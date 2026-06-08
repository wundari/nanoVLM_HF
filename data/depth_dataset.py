"""
depth_dataset.py
================
nanoVLM-compatible dataloader for:
  AdamYao/3D_Visual_Illusion_Depth_Estimation

Dataset layout (WebDataset / tar.gz)
-------------------------------------
Each row in the HuggingFace dataset has:
  - png       : raw image bytes  (PIL-loadable)
  - __key__   : path string, e.g.
                  "fooling-3d_2/left/video10/SceneName/frame_0634"
                  "fooling-3d_2/depth/video10/SceneName/frame_0634"
                  "fooling-3d_2/right/video10/SceneName/frame_0634"
                  "fooling-3d_2/mask/video10/SceneName/frame_0634"

Strategy
--------
1. Stream the dataset once, routing each row into a dict keyed by its
   canonical frame ID (everything after the modality folder).
2. Yield (left_image, depth_image, scene_name) triples whenever both
   modalities are available for the same frame.
3. Wrap in nanoVLM's VQADataset interface so it plugs directly into
   the existing train.py collation and training loop.

Usage
-----
    from depth_dataset import build_depth_dataloader
    from models.config import VLMConfig
    from data.processors import get_tokenizer, get_image_processor
    from models.vision_language_model import VisionLanguageModel

    cfg = VLMConfig()
    tokenizer      = get_tokenizer(cfg.lm_tokenizer)
    image_processor = get_image_processor(cfg.vit_img_size)

    train_loader, val_loader = build_depth_dataloader(
        cfg, tokenizer, image_processor,
        val_split=0.05,   # 5 % of frames go to validation
        batch_size=4,
        num_workers=4,
    )

    for batch in train_loader:
        # batch has keys: images, input_ids, attention_mask, labels
        # — identical to nanoVLM's default VQADataset output
        ...
"""

import io
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ── nanoVLM imports (adjust path if you run from outside the repo root) ──────
from data.processors import get_image_string, get_tokenizer, get_image_processor
from data.collators import VQACollator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  QA PROMPT TEMPLATES
#  Feel free to extend / replace these with domain-specific questions.
# ─────────────────────────────────────────────────────────────────────────────

QA_TEMPLATES: List[Dict[str, str]] = [
    {
        "user": (
            "This image comes from a 3D visual illusion scene. "
            "Describe the apparent depth structure: which elements appear "
            "closest to the viewer and which seem furthest away?"
        ),
        "assistant": (
            "The scene displays a 3D visual illusion where depth cues create "
            "the impression of spatial layering. Foreground elements appear "
            "prominently close to the viewer, while background regions recede "
            "into the distance, creating the characteristic illusion of depth."
        ),
    },
    {
        "user": (
            "Look at this frame from a 3D visual illusion video. "
            "Does the scene create a convincing sense of depth? "
            "What visual cues contribute to the illusion?"
        ),
        "assistant": (
            "Yes, the scene leverages several depth cues to produce a "
            "convincing 3D illusion: perspective lines converging toward a "
            "vanishing point, relative object sizes indicating distance, "
            "and shading gradients that reinforce the sense of spatial relief."
        ),
    },
    {
        "user": (
            "Estimate the relative depth ordering in this visual illusion image. "
            "Identify the near, mid, and far regions."
        ),
        "assistant": (
            "Based on the visual information: the near region consists of "
            "prominently lit or large-appearing elements in the foreground. "
            "The mid-ground contains transitional elements at moderate depth. "
            "The far region shows smaller, lower-contrast elements that "
            "recede toward the background of the illusion."
        ),
    },
    {
        "user": (
            "This is a stereo-recorded 3D visual illusion scene. "
            "What aspects of the image make it appear three-dimensional?"
        ),
        "assistant": (
            "The three-dimensional appearance stems from multiple visual "
            "mechanisms: binocular disparity encoded in the stereo pair, "
            "object occlusion patterns, surface texture gradients, and "
            "cast shadows that anchor objects in the perceived 3D space."
        ),
    },
    {
        "user": (
            "Analyze the depth map information in this scene. "
            "Where are the closest and farthest points from the camera?"
        ),
        "assistant": (
            "The closest points to the camera are typically the largest, "
            "sharpest, or most brightly illuminated elements in the frame. "
            "The farthest points correspond to smaller, hazier, or more "
            "darkly rendered regions. The depth gradient transitions smoothly "
            "across the illusion, creating a continuous sense of 3D space."
        ),
    },
]


def _random_qa() -> Dict[str, str]:
    """Pick a random QA template for a given sample."""
    return random.choice(QA_TEMPLATES)


# ─────────────────────────────────────────────────────────────────────────────
#  KEY PARSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def _parse_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a dataset __key__ string into (modality, frame_id).

    Example
    -------
    "fooling-3d_2/left/video10/SceneName/frame_0634"
        → modality  = "left"
          frame_id  = "video10/SceneName/frame_0634"

    Returns (None, None) if the key does not match the expected pattern.
    """
    parts = key.split("/")
    # Expected: [dataset_part, modality, video*, scene, frame]
    # Minimum length to be useful: 4 parts
    if len(parts) < 4:
        return None, None

    modality = parts[1]  # "left" | "right" | "depth" | "mask"
    frame_id = "/".join(parts[2:])  # "video10/SceneName/frame_0634"
    return modality, frame_id


def _extract_scene_name(frame_id: str) -> str:
    """Return the human-readable scene / video name from a frame_id."""
    parts = frame_id.split("/")
    # frame_id = "video10/SceneName/frame_0634" → "SceneName"
    if len(parts) >= 2:
        return parts[-2]
    return frame_id


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME PAIRING LOGIC
# ─────────────────────────────────────────────────────────────────────────────


def stream_paired_frames(
    hf_dataset,
    required_modalities: Tuple[str, ...] = ("left", "depth"),
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Stream a HuggingFace dataset (WebDataset-converted) and return a list of
    dicts where every requested modality is present for the same frame_id.

    Parameters
    ----------
    hf_dataset       : HuggingFace Dataset object (iterable or map-style)
    required_modalities : which image types must be present for a valid pair
    max_samples      : cap the output (useful for quick experiments)

    Returns
    -------
    List of dicts:
        {
          "left":     PIL.Image,
          "depth":    PIL.Image,   (if requested)
          "right":    PIL.Image,   (if requested)
          "mask":     PIL.Image,   (if requested)
          "frame_id": str,
          "scene":    str,
        }
    """
    # Buffer: frame_id → {modality: PIL.Image}
    buffer: Dict[str, Dict[str, Image.Image]] = defaultdict(dict)
    paired: List[Dict] = []

    logger.info("Streaming dataset to pair frames by frame_id …")

    for row in hf_dataset:
        key = row.get("__key__", "")
        modality, frame_id = _parse_key(key)

        if modality is None or modality not in required_modalities:
            continue

        # Decode bytes → PIL Image
        img_bytes = row.get("png")
        if img_bytes is None:
            continue
        try:
            if isinstance(img_bytes, bytes):
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                # Some HF decodings give a PIL Image directly
                img = img_bytes.convert("RGB")
        except Exception as e:
            logger.warning(f"Could not decode image for key={key}: {e}")
            continue

        buffer[frame_id][modality] = img

        # Check completeness
        if all(m in buffer[frame_id] for m in required_modalities):
            entry = dict(buffer.pop(frame_id))
            entry["frame_id"] = frame_id
            entry["scene"] = _extract_scene_name(frame_id)
            paired.append(entry)

            if max_samples and len(paired) >= max_samples:
                logger.info(f"Reached max_samples={max_samples}, stopping stream.")
                break

    logger.info(f"Paired {len(paired)} complete frame sets.")
    return paired


# ─────────────────────────────────────────────────────────────────────────────
#  NANOVLM-COMPATIBLE DATASET
# ─────────────────────────────────────────────────────────────────────────────


class DepthEstimationVQADataset(Dataset):
    """
    Wraps the 3D Visual Illusion dataset in the exact format that nanoVLM's
    VQADataset.BaseDataset expects:

        item = {
            "images": List[PIL.Image],          # one or more images
            "texts":  List[{"user": str,        # list of QA turns
                             "assistant": str}],
        }

    The left RGB frame is always provided as the primary image.
    Optionally, the depth map is appended as a second image so the model
    sees both views (set include_depth_image=True).
    """

    def __init__(
        self,
        paired_frames: List[Dict],
        tokenizer,
        image_processor,
        mp_image_token_length: int,
        include_depth_image: bool = False,
        seed: int = 42,
    ):
        self.frames = paired_frames
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.include_depth = include_depth_image

        # Build the internal VQADataset using nanoVLM's BaseDataset
        # We convert our frames into the expected HF-style dict format
        self._vqa = _NanoVLMAdapter(
            frames=paired_frames,
            tokenizer=tokenizer,
            image_processor=image_processor,
            mp_image_token_length=mp_image_token_length,
            include_depth=include_depth_image,
        )

    def __len__(self):
        return len(self._vqa)

    def __getitem__(self, idx):
        return self._vqa[idx]


class _NanoVLMAdapter(Dataset):
    """
    Internal adapter that converts our paired_frames list into the dict
    format that BaseDataset._process_data expects, and handles the full
    tokenisation + label-masking pipeline.

    Inherits all the image processing and tokenisation logic from
    BaseDataset by composition rather than inheritance to keep things
    explicit and easy to modify.
    """

    def __init__(
        self,
        frames,
        tokenizer,
        image_processor,
        mp_image_token_length,
        include_depth=False,
    ):
        from data.datasets import BaseDataset  # nanoVLM's BaseDataset

        self.frames = frames
        self.include_depth = include_depth

        # Re-use BaseDataset's processing helpers by creating a thin wrapper
        # dataset object that yields items in the expected format.
        wrapped_hf_dataset = _FrameListWrapper(frames, include_depth)
        self.base = BaseDataset(
            dataset=wrapped_hf_dataset,
            tokenizer=tokenizer,
            image_processor=image_processor,
            mp_image_token_length=mp_image_token_length,
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        from data.datasets import VQADataset  # re-use _process_data logic

        item = self.base.dataset[idx]  # get the raw formatted item
        return self.base._process_data_from_item(item)  # see wrapper below


class _FrameListWrapper:
    """
    Presents our list of paired_frames as an iterable/indexable object
    that yields items in the exact schema that BaseDataset._get_messages
    and VQADataset._process_data expect:

        {
          "images": [PIL.Image, ...],
          "texts":  [{"user": str, "assistant": str}],
        }
    """

    def __init__(self, frames: List[Dict], include_depth: bool = False):
        self.frames = frames
        self.include_depth = include_depth

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx) -> Dict:
        frame = self.frames[idx]
        left_img = frame["left"]
        depth_img = frame.get("depth")
        scene = frame.get("scene", "unknown scene")

        # Build image list
        images = [left_img]
        if self.include_depth and depth_img is not None:
            images.append(depth_img)

        # Pick a QA template and personalise with scene name
        qa = _random_qa()
        user_text = f"Scene: {scene}. " + qa["user"]
        assistant_text = qa["assistant"]

        return {
            "images": images,
            "texts": [{"user": user_text, "assistant": assistant_text}],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE DATASET  (does NOT depend on nanoVLM's BaseDataset)
#  Use this if you run outside the nanoVLM repo or want more control.
# ─────────────────────────────────────────────────────────────────────────────


class StandaloneDepthDataset(Dataset):
    """
    Self-contained nanoVLM-compatible Dataset that reproduces the
    tokenisation and label-masking pipeline of BaseDataset / VQADataset
    without importing from nanoVLM.  Useful for debugging or standalone use.

    Output dict per sample:
        {
          "images":         List[Tensor(n_patches, C, H, W)],
          "input_ids":      LongTensor,
          "attention_mask": LongTensor,
          "labels":         LongTensor,  (-100 on non-answer tokens)
        }
    """

    def __init__(
        self,
        paired_frames: List[Dict],
        tokenizer,
        image_processor,
        mp_image_token_length: int,
        include_depth_image: bool = False,
    ):
        self.frames = paired_frames
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.include_depth = include_depth_image

        # Compute prefix_len once (skips "assistant\n" in loss mask)
        self.prefix_len = self._compute_prefix_len()

    # ── prefix measurement (same logic as BaseDataset._get_prefix_len) ──────

    def _compute_prefix_len(self) -> int:
        probe = "xzyvd"
        templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": probe}],
            tokenize=False,
            add_special_tokens=False,
        )
        loc = templated.find(probe)
        return len(self.tokenizer.encode(templated[:loc]))

    # ── image processing ─────────────────────────────────────────────────────

    def _process_image(self, img: Image.Image):
        """Run DynamicResize + GlobalAndSplitImages. Returns (tensor, (rows,cols))."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        processed, split_count = self.image_processor(img)
        # strip global view if tokenizer has no <global_image> token
        if (
            not hasattr(self.tokenizer, "global_image_token")
            and split_count[0] * split_count[1] == len(processed) - 1
        ):
            processed = processed[1:]
        return processed, split_count

    # ── message construction ─────────────────────────────────────────────────

    def _build_messages(self, frame: Dict) -> Tuple[List[Dict], List]:
        left_img = frame["left"]
        depth_img = frame.get("depth")
        scene = frame.get("scene", "")
        qa = _random_qa()

        # Process images
        images_data = [left_img]
        if self.include_depth and depth_img is not None:
            images_data.append(depth_img)

        processed_images = []
        split_counts = []
        for img in images_data:
            tensor, count = self._process_image(img)
            processed_images.append(tensor)
            split_counts.append(count)

        # Build image string placeholder
        image_string = get_image_string(
            self.tokenizer, split_counts, self.mp_image_token_length
        )

        user_text = f"Scene: {scene}. " + qa["user"]
        messages = [
            {"role": "user", "content": image_string + user_text},
            {"role": "assistant", "content": qa["assistant"]},
        ]

        return messages, processed_images

    # ── tokenisation + loss mask ──────────────────────────────────────────────

    def _tokenise_and_mask(self, messages: List[Dict]):
        # Tokenise full conversation
        conv = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False, return_dict=True
        )
        input_ids = conv["input_ids"]
        attn_mask = conv["attention_mask"]

        # Build loss mask: 1 only on assistant answer tokens
        mask = [0] * len(input_ids)
        cursor = 0
        for msg in messages:
            seg = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(seg)
            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)
            cursor += seg_len

        return (
            torch.tensor(input_ids),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(attn_mask),
        )

    # ── label construction (causal shift) ────────────────────────────────────

    @staticmethod
    def _build_labels(input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)
        labels[-1] = -100
        return labels

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        try:
            messages, processed_images = self._build_messages(frame)
            input_ids, mask, attn_mask = self._tokenise_and_mask(messages)
            labels = self._build_labels(input_ids, mask)

            return {
                "images": processed_images,
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels,
            }
        except Exception as e:
            logger.warning(f"Skipping frame {frame.get('frame_id')}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  COLLATOR  (mirrors nanoVLM's VQACollator)
# ─────────────────────────────────────────────────────────────────────────────


def depth_collate_fn(batch, pad_token_id: int):
    """
    Collate a list of samples (each from StandaloneDepthDataset.__getitem__)
    into a batched dict.  Left-pads input_ids and attention_mask; pads labels
    with -100 so cross-entropy ignores padding.

    Filters out None samples (failed items).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Images: each sample has a list of patch tensors; combine into one list
    all_images = []
    for sample in batch:
        all_images.extend(sample["images"])  # list of Tensor(n_patches,C,H,W)

    # Determine max sequence length for padding
    max_len = max(s["input_ids"].size(0) for s in batch)

    def left_pad(tensor: torch.Tensor, pad_val: int) -> torch.Tensor:
        pad_size = max_len - tensor.size(0)
        return torch.nn.functional.pad(tensor, (pad_size, 0), value=pad_val)

    input_ids = torch.stack([left_pad(s["input_ids"], pad_token_id) for s in batch])
    attention_mask = torch.stack([left_pad(s["attention_mask"], 0) for s in batch])
    labels = torch.stack([left_pad(s["labels"], -100) for s in batch])

    return {
        "images": all_images,  # List[Tensor] — handled by VLM.forward
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def build_depth_dataloader(
    cfg,
    tokenizer,
    image_processor,
    *,
    val_split: float = 0.05,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    include_depth_image: bool = False,
    seed: int = 42,
    streaming: bool = True,
):
    """
    Load AdamYao/3D_Visual_Illusion_Depth_Estimation, pair left+depth frames,
    and return (train_loader, val_loader) that are drop-in replacements for
    nanoVLM's default dataloaders.

    Parameters
    ----------
    cfg                 : VLMConfig (needs cfg.mp_image_token_length)
    tokenizer           : from get_tokenizer(cfg.lm_tokenizer)
    image_processor     : from get_image_processor(cfg.vit_img_size)
    val_split           : fraction of paired frames reserved for validation
    batch_size          : per-GPU batch size
    num_workers         : DataLoader workers
    max_samples         : cap total paired frames (None = use all)
    include_depth_image : if True, depth map is passed as a second image
    seed                : random seed for split reproducibility
    streaming           : use HF streaming (recommended for 456 GB dataset)

    Returns
    -------
    (train_loader, val_loader)  — torch DataLoader objects
    """

    # 1. Load the HuggingFace dataset ─────────────────────────────────────────
    logger.info("Loading AdamYao/3D_Visual_Illusion_Depth_Estimation …")
    hf_ds = load_dataset(
        "AdamYao/3D_Visual_Illusion_Depth_Estimation",
        split="train",
        streaming=streaming,
    )

    # 2. Pair left + depth frames ─────────────────────────────────────────────
    paired = stream_paired_frames(
        hf_ds,
        required_modalities=("left", "depth"),
        max_samples=max_samples,
    )

    if not paired:
        raise RuntimeError(
            "No paired frames found. Check that the dataset downloaded correctly "
            "and that __key__ fields contain 'left' and 'depth' modality paths."
        )

    # 3. Train / val split ─────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(paired)
    n_val = max(1, int(len(paired) * val_split))
    val_frames = paired[:n_val]
    train_frames = paired[n_val:]

    logger.info(f"Split: {len(train_frames)} train / {len(val_frames)} val frames")

    # 4. Build Dataset objects ─────────────────────────────────────────────────
    train_ds = StandaloneDepthDataset(
        paired_frames=train_frames,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=cfg.mp_image_token_length,
        include_depth_image=include_depth_image,
    )
    val_ds = StandaloneDepthDataset(
        paired_frames=val_frames,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=cfg.mp_image_token_length,
        include_depth_image=include_depth_image,
    )

    # 5. Collate function ──────────────────────────────────────────────────────
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        return depth_collate_fn(batch, pad_id)

    # 6. DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK SMOKE TEST  (run: python depth_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # ── Minimal mock objects so you can test the pipeline offline ─────────────
    class _MockTokenizer:
        """Bare-minimum tokenizer stub for offline testing."""

        image_token = "<|image|>"
        image_token_id = 99
        global_image_token = "<global_image>"
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text):
            return list(range(min(len(text), 20)))

        def apply_chat_template(
            self, messages, tokenize=False, add_special_tokens=False, return_dict=False
        ):
            text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
            if not tokenize:
                return text
            ids = list(range(min(len(text), 64)))
            if return_dict:
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
            return ids

    class _MockImageProcessor:
        """Returns a single 3×64×64 patch tensor per image."""

        def __call__(self, img):
            import torch

            tensor = torch.zeros(1, 3, 64, 64)  # (1 patch, C, H, W)
            return tensor, (1, 1)  # split_count = (rows=1, cols=1)

    # ── Create two fake paired frames ─────────────────────────────────────────
    fake_frame = {
        "left": Image.new("RGB", (64, 64), color=(100, 150, 200)),
        "depth": Image.new("RGB", (64, 64), color=(30, 30, 30)),
        "frame_id": "video0/TestScene/frame_0001",
        "scene": "TestScene",
    }
    paired = [fake_frame, fake_frame]

    tokenizer = _MockTokenizer()
    image_processor = _MockImageProcessor()

    ds = StandaloneDepthDataset(
        paired_frames=paired,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=4,
        include_depth_image=False,
    )

    sample = ds[0]
    print("\n✅ StandaloneDepthDataset smoke test passed")
    print(f"   images:         {len(sample['images'])} patch tensor(s)")
    print(f"   input_ids:      shape {sample['input_ids'].shape}")
    print(f"   attention_mask: shape {sample['attention_mask'].shape}")
    print(f"   labels:         shape {sample['labels'].shape}")
    print(f"   labels (non -100): {(sample['labels'] != -100).sum().item()} tokens")

    # ── Test key parsing ──────────────────────────────────────────────────────
    tests = [
        (
            "fooling-3d_2/left/video10/SceneName/frame_0634",
            ("left", "video10/SceneName/frame_0634"),
        ),
        (
            "fooling-3d_2/depth/video10/SceneName/frame_0634",
            ("depth", "video10/SceneName/frame_0634"),
        ),
        ("fooling3D/right/video1/X/frame_0001", ("right", "video1/X/frame_0001")),
        ("bad/key", (None, None)),
    ]
    for key, expected in tests:
        got = _parse_key(key)
        status = "✅" if got == expected else "❌"
        print(f"   {status} _parse_key({key!r}) = {got}")
