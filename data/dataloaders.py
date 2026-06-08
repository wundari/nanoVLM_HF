import torch
from torch.utils.data import DataLoader

from data.collators import VQACollator


def build_vqa_dataset(raw_dataset, cfg, tokenizer=None, image_processor=None):
    """Wrap raw VQA records with image/text preprocessing.

    Conceptually copied from upstream nanoVLM's train.py data setup, but kept
    small for the from-scratch path: this function accepts an already-created
    raw dataset instead of downloading/loading one itself.
    """

    from data.datasets import VQADataset
    from data.processors import get_image_processor, get_tokenizer

    tokenizer = tokenizer or get_tokenizer(
        cfg.lm_tokenizer,
        cfg.vlm_extra_tokens,
        cfg.lm_chat_template,
    )
    image_processor = image_processor or get_image_processor(
        cfg.max_img_size,
        cfg.vit_img_size,
        cfg.resize_to_max_side_len,
    )

    return VQADataset(
        raw_dataset,
        tokenizer,
        image_processor,
        cfg.mp_image_token_length,
    )


def build_vqa_dataloader(
    dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=False,
    num_workers=0,
    drop_last=False,
):
    """Build the first simple VQA DataLoader.

    Raw input:
        dataset: VQADataset or any dataset returning VQADataset-shaped samples
        tokenizer: tokenizer with pad_token_id
        batch_size: number of samples per batch
        max_length: padded sequence length

    Expected batch:
        input_ids: [B, max_length]
        attention_mask: [B, max_length]
        labels: [B, max_length]
        images: list length B
    """

    generator = torch.Generator()
    generator.manual_seed(0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=VQACollator(tokenizer, max_length),
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
    )
