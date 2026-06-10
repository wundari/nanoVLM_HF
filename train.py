# %%
import __main__
import math
import numpy as np
import random
import time
import wandb
import contextlib
import subprocess
import json
import re
import glob

from statistics import mean
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from models.vision_language_model import VisionLanguageModel

from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step

from configs.config import VLMConfig, TrainConfig

from datasets import (
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_from_disk,
)
from datetime import timedelta
from jaxtyping import Float

# Otherwise, the tokenizer will throw a warning
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HTTP_PROXY"] = "http://proxy.nict.go.jp:3128"
os.environ["HTTPS_PROXY"] = "http://proxy.nict.go.jp:3128"


PG_CPU = None


# %% auxiliary functions
def init_dist():
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def destroy_dist():
    dist.destroy_process_group()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_run_name(cfg_train: TrainConfig, vlm_cfg: VLMConfig):
    batch_size = f"bs{int(cfg_train.batch_size * get_world_size() * cfg_train.gradient_accumulation_steps)}"
    max_training_steps = f"{cfg_train.max_training_steps}"
    learning_rate = f"lr_vision_{cfg_train.lr_vision_backbone}-language_{cfg_train.lr_language_backbone}-{cfg_train.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"


def dist_mean_scalar(x: float | int) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)

    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)  # in‑place, returns None
    t /= dist.get_world_size()
    return t.item()


def wrap_model(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    return DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )


def dist_gather(obj):
    """
    Gather *any* picklable object from every rank without allocating
    temporary CUDA buffers.  Returns a list [rank0_obj, rank1_obj, …].

    Falls back to a single-rank list when torch.distributed is not initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)  # CPU path
    return result


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# %% set up dataloader

# cfg_train = TrainConfig()
# cfg_vlm = VLMConfig()


def build_dataloaders(cfg_train: TrainConfig, cfg_vlm: VLMConfig):

    print(f"build dataloaders from {cfg_train.train_dataset_path}")

    # create datasets
    tokenizer = get_tokenizer(
        name=cfg_vlm.lm_tokenizer,
        extra_special_tokens=cfg_vlm.vlm_extra_tokens,
        chat_template=cfg_vlm.lm_chat_template,
    )
    image_processor = get_image_processor(
        max_img_size=cfg_vlm.max_img_size,  # 2048
        splitted_image_size=cfg_vlm.vit_img_size,  # 512
        resize_to_max_side_len=cfg_vlm.resize_to_max_side_len,
    )

    if cfg_train.use_local_dataset:
        ## load local dataset
        # detect if train_dataset_path is a local directory containing parquets files
        dataset_names_to_load = cfg_train.train_dataset_name
        local_parquet_files = []
        if "all" in dataset_names_to_load:
            complete_names = get_dataset_config_names(
                f"HuggingFaceM4/{cfg_train.dataset_name}"
            )
            dataset_names_to_load = [
                p.name
                for p in Path(cfg_train.train_dataset_path).iterdir()
                if (p.is_dir() and p.name in complete_names)
            ]
        for dataset_name in dataset_names_to_load:
            parquet_files = sorted(
                glob.glob(
                    os.path.join(
                        cfg_train.train_dataset_path, dataset_name, "*.parquet"
                    )
                )
            )
            local_parquet_files.append(parquet_files)

        local_parquet_files = sorted(
            [name for subdataset in local_parquet_files for name in subdataset]
        )

        ## local dataset
        print(f"Detected local parquet files in {cfg_train.train_dataset_path}")
        print(
            f"Found {len(local_parquet_files)} parquet shards. Loading from local drive..."
        )
        dataset_train = load_dataset(
            "parquet",
            data_files={"train": local_parquet_files},
            split="train",
            num_proc=4,
            keep_in_memory=False,
        )
        print(f"Loaded local dataset with {len(dataset_train)} samples.")

    ## online dataset
    else:
        dataset_names_to_load = cfg_train.train_dataset_name
        if "shards" in cfg_train.train_dataset_name:
            print("Loading shards")
            total_shards = 56
            dataset_names_to_load = [
                cfg_train.train_dataset_path + f"/shard_{i}"
                for i in range(total_shards)
            ]
        if "all" in dataset_names_to_load:
            dataset_names_to_load = get_dataset_config_names(
                cfg_train.train_dataset_path
            )

        # load and combine all training datasets
        combined_train_data = []
        for dataset_name in dataset_names_to_load:
            print(f"Loading dataset: {dataset_name}")

            if "shard_" in dataset_name:
                try:
                    dataset_train = load_from_disk(dataset_name)
                    combined_train_data.append(dataset_train)
                    continue
                except Exception as e:
                    print(
                        f"Warning: failed to load dataset shard {dataset_name} from {cfg_train.train_dataset_path}. Error: {e}"
                    )
                    continue
            try:
                dataset_train = load_dataset(
                    cfg_train.train_dataset_path,
                    dataset_name,
                    streaming=cfg_train.stream_dataset,
                    on_bad_files="warn",
                    num_proc=4 if not cfg_train.stream_dataset else None,
                )["train"]
                if cfg_train.stream_dataset:
                    next(iter(dataset_train))  # check if dataset is loaded correctly
                combined_train_data.append(dataset_train)
            except Exception as e:
                if is_master():
                    print(
                        f"Warning: Failed to load dataset config {dataset_name} from {cfg_train.train_dataset_path}. Error: {e}"
                    )
                continue

        if not combined_train_data:
            raise ValueError(
                "No valid datasets were loaded. Please check your dataset path and configurations."
            )

        dataset_train = concatenate_datasets(combined_train_data)

    if not cfg_train.stream_dataset:
        # Shuffle the training dataset,
        # so train and val get equal contributions from all concatenated datasets
        dataset_train = dataset_train.shuffle(seed=0)

    if is_dist():
        # We need to shard the dataset in DDP since we are using an iterable dataset
        # instead of the distributed sampler
        dataset_train = dataset_train.shard(
            num_shards=get_world_size(), index=get_rank()
        )

    val_size = int(cfg_train.val_size / get_world_size())
    print(f"Val size per GPU: {val_size}")

    if cfg_train.stream_dataset:
        dataset_val = dataset_train.take(val_size)
        dataset_train = dataset_train.skip(val_size)
    else:
        dataset_val = dataset_train.select(range(val_size))
        dataset_train = dataset_train.select(range(val_size, len(dataset_train)))

    dataset_vqa_train = VQADataset(
        dataset_train,
        tokenizer,
        image_processor,
        cfg_vlm.mp_image_token_length,
        cfg_train.relevance_min_rating,
        cfg_train.image_correspondence_min_rating,
        cfg_train.visual_dependency_min_rating,
        cfg_train.formatting_min_rating,
    )
    dataset_vqa_val = VQADataset(
        dataset_val,
        tokenizer,
        image_processor,
        cfg_vlm.mp_image_token_length,
        cfg_train.relevance_min_rating,
        cfg_train.image_correspondence_min_rating,
        cfg_train.visual_dependency_min_rating,
        cfg_train.formatting_min_rating,
    )

    effective_max_sample = min(
        cfg_train.max_sample_length,
        cfg_vlm.lm_max_length,  # hard upper bound = model context window
    )
    dataset_vqa_fixed_train = ConstantLengthDataset(
        dataset_vqa_train,
        infinite=False,
        max_sample_length=effective_max_sample,
        seq_length=cfg_vlm.lm_max_length,
        num_of_sequences=cfg_train.batch_size * 4,
        queue_size=8,
        max_images_per_example=cfg_train.max_images_per_example,
        max_images_per_knapsack=cfg_train.max_images_per_knapsack,
    )

    dataset_vqa_fixed_val = ConstantLengthDataset(
        dataset_vqa_val,
        infinite=False,
        max_sample_length=effective_max_sample,
        seq_length=cfg_vlm.lm_max_length,
        num_of_sequences=cfg_train.batch_size * 4,
        queue_size=8,
        max_images_per_example=cfg_train.max_images_per_example,
        max_images_per_knapsack=cfg_train.max_images_per_knapsack,
    )

    # create collaotrs
    vqa_collator = VQACollator(tokenizer, cfg_vlm.lm_max_length)

    g = torch.Generator()
    g.manual_seed(0)

    # create dataloaders
    train_loader = DataLoader(
        dataset_vqa_fixed_train,
        batch_size=cfg_train.batch_size,
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        dataset_vqa_fixed_val,
        batch_size=cfg_train.batch_size,
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # warmup dataloaders to kickstart worker processes
    print("Warming up dataloaders")
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    next(iter_train_loader)
    next(iter_val_loader)
    print("Warmup complete.")

    return train_loader, val_loader, iter_train_loader, iter_val_loader


# %% debug dataset
# batch_raw = dataset_train[0]
# images_raw = batch_raw["images"]
# np.shape(images_raw)  # [1, 257, 800, 3]

# batch_vqa = dataset_vqa_train[0]
# images_vqa = batch_vqa["images"]
# np.shape(images_vqa)  # [1, 9, 3, 512, 512]

# batch_vqa_fixed = next(iter(dataset_vqa_fixed_train))
# input_ids_vqa_fixed = batch_vqa_fixed["input_ids"]
# attention_mask_vqa_fixed = batch_vqa_fixed["attention_mask"]
# images_vqa_fixed = batch_vqa_fixed["images"]
# np.shape(images_vqa_fixed)  # [1, 13, 3, 512, 512]

# batch_loader = next(iter(train_loader))
# input_ids_loader = batch_loader["input_ids"]
# images_loader = batch_loader["images"]
# np.shape(images_loader[0])  # [1, 17, 3, 512, 512]


# %%


def get_lr(it, max_lr, max_steps):
    """
    # Cosine learning rate schedule with warmup (from Karpathy)
    # https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353

    """
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# %%

# cfg_train = TrainConfig()
# cfg_vlm = VLMConfig()


def train(cfg_train: TrainConfig, cfg_vlm: VLMConfig):

    train_loader, val_loader, iter_train_loader, iter_val_loader = build_dataloaders(
        cfg_train, cfg_vlm
    )

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = get_run_name(cfg_train, cfg_vlm)
    if cfg_train.log_wandb and is_master():
        run = wandb.init(
            entity=cfg_train.wandb_entity,
            project="nanoVLM",
            config={"VLMConfig": asdict(cfg_vlm), "TrainConfig": asdict(cfg_train)},
            name=run_name,
        )

        # define a custom x-axis for lmms-eval metrics
        lmms_eval_step = "<lmms-eval-step>"
        run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

    # initialize model
    if cfg_train.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {cfg_vlm.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(cfg_vlm.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(
            cfg_vlm, load_backbone=cfg_vlm.vlm_load_backbone_weights
        )

    ## define optimizer groups
    # Since we have pretrained vision and language backbones,
    # but a newly initialized modality projection layer,
    # it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer,
    # but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = []
    if cfg_train.lr_mp > 0:
        param_groups.append(
            {"params": list(model.MP.parameters()), "lr": cfg_train.lr_mp}
        )
    else:
        for p in list(model.MP.parameters()):
            p.requires_grad = False

    if cfg_train.lr_vision_backbone > 0:
        param_groups.append(
            {
                "params": list(model.vision_encoder.parameters()),
                "lr": cfg_train.lr_vision_backbone,
            }
        )
    else:
        for p in list(model.vision_encoder.parameters()):
            p.requires_grad = False

    if cfg_train.lr_language_backbone > 0:
        param_groups.append(
            {
                "params": list(model.decoder.parameters()),
                "lr": cfg_train.lr_language_backbone,
            }
        )
    else:
        for p in list(model.decoder.parameters()):
            p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    device = torch.device(cfg_train.device)
    print(f"Using device: {device}")
    model.to(device)

    if cfg_train.compile:
        model = torch.compile(model, mode=cfg_train.compile_mode, dynamic=False)
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float("inf")
    best_model_path = None
    logged_eval_steps = set()
    global_step = 0
    epoch = 0

    # training stats accumulator
    accumulated_stats = {
        "tokens_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
    }

    while global_step < cfg_train.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        print("Starting training loop")
        for i, batch in enumerate(
            synchronized_dataloader_step(iter_train_loader, is_dist())
        ):
            is_update_step = (i + 1) % cfg_train.gradient_accumulation_steps == 0
            batch_start_time = time.time()
            images = batch[
                "images"
            ]  # [dataloader_batch_size, original_images_inside_packed_sequence, n_tiles, rgb_channels, H, W]
            input_ids = batch["input_ids"].to(
                device
            )  # [dataloader_batch_size, max_seq_length=VLMConfig.lm_max_length]
            labels = batch["labels"].to(
                device
            )  # [dataloader_batch_size, max_seq_length=VLMConfig.lm_max_length]
            attention_mask = batch["attention_mask"].to(
                device
            )  # [dataloader_batch_size, max_seq_length=VLMConfig.lm_max_length]
            data_load_time = time.time() - data_load_start

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (
                is_dist()
                and cfg_train.gradient_accumulation_steps > 1
                and not is_update_step
            ):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
            )
            with autocast_context:
                with context:
                    _, loss = model(
                        input_ids, images, attention_mask=attention_mask, targets=labels
                    )

            if cfg_train.gradient_accumulation_steps > 1:
                loss = loss / cfg_train.gradient_accumulation_steps

            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                if cfg_train.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, max_norm=cfg_train.max_grad_norm
                    )

                param_group_idx = 0
                if cfg_train.lr_mp > 0:
                    adj_lr_mp = get_lr(
                        global_step, cfg_train.lr_mp, cfg_train.max_training_steps
                    )
                    optimizer.param_groups[param_group_idx]["lr"] = adj_lr_mp
                    param_group_idx += 1

                if cfg_train.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_lr(
                        global_step,
                        cfg_train.lr_vision_backbone,
                        cfg_train.max_training_steps,
                    )
                    optimizer.param_groups[param_group_idx][
                        "lr"
                    ] = adj_lr_vision_backbone
                    param_group_idx += 1

                if cfg_train.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_lr(
                        global_step,
                        cfg_train.lr_language_backbone,
                        cfg_train.max_training_steps,
                    )
                    optimizer.param_groups[param_group_idx][
                        "lr"
                    ] = adj_lr_language_backbone

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if cfg_train.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * cfg_train.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(
                attention_mask
            ).item()  # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = (
                get_world_size() * num_tokens / batch_duration
            )  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats["tokens_per_second"].append(tokens_per_second)
            accumulated_stats["data_load_time"].append(data_load_time)
            accumulated_stats["fw_bw_time"].append(fw_bw_time)
            accumulated_stats["post_process_time"].append(post_process_time)
            accumulated_stats["images_per_sample"].extend(images_per_sample)

            if (
                cfg_train.eval_in_epochs
                and global_step % cfg_train.eval_interval == 0
                and is_update_step
            ):
                print("Starting evaluation")
                model.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    val_batches = 0
                    for batch in synchronized_dataloader_step(
                        iter_val_loader, is_dist()
                    ):
                        if val_batches > 64:
                            print(f"Evaluated {val_batches} batches")
                            break
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(
                                input_ids,
                                images,
                                attention_mask=attention_mask,
                                targets=labels,
                            )

                        total_val_loss += loss.item()
                        val_batches += 1

                    iter_val_loader = iter(val_loader)
                    avg_val_loss = (
                        total_val_loss / val_batches if val_batches > 0 else 0
                    )
                    avg_val_loss = (
                        mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    )

                    checkpoint_path_step = ""
                    if is_master():
                        # Save a checkpoint for this evaluation step
                        checkpoint_path_step = os.path.join(
                            cfg_vlm.vlm_checkpoint_path, run_name, f"step_{global_step}"
                        )
                        save_model = (
                            model.module if is_dist() else model
                        )  # unwrap the model for saving if DDP
                        save_model.save_pretrained(save_directory=checkpoint_path_step)

                        if (
                            cfg_train.use_lmms_eval
                            and global_step % (cfg_train.eval_interval * 2) == 0
                        ):
                            # Submit evaluation job
                            cmd = f"sbatch eval.slurm {checkpoint_path_step} {global_step} {run_name} {cfg_train.lmms_eval_limit} {cfg_train.lmms_eval_tasks} {cfg_train.lmms_eval_batch_size}"
                            print(f"Submitting evaluation job: {cmd}")
                            subprocess.run(cmd, shell=True)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            best_model_path = checkpoint_path_step

                    if is_master():
                        print(
                            f"Step: {global_step}/{cfg_train.max_training_steps}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}"
                        )
                        if cfg_train.log_wandb:
                            run.log({"val_loss": avg_val_loss}, step=global_step)

                model.train()

            # Log training stats every N steps (ALL RANKS must participate in collective ops)
            if (
                global_step % cfg_train.stats_log_interval == 0
                and len(accumulated_stats["tokens_per_second"]) > 0
                and is_update_step
            ):
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in [
                    "tokens_per_second",
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]  # Flatten list of lists
                        stats[f"avg_{key}"] = mean(all_values_flat)
                    else:
                        stats[f"avg_{key}"] = mean(accumulated_stats[key])

                for key in [
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]
                        stats[f"max_{key}"] = max(all_values_flat)
                    else:
                        stats[f"max_{key}"] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(
                        accumulated_stats["images_per_sample"]
                    )
                    all_images_flat = [
                        item for sublist in all_images_values for item in sublist
                    ]
                    stats["min_images_per_sample"] = min(all_images_flat)
                else:
                    stats["min_images_per_sample"] = min(
                        accumulated_stats["images_per_sample"]
                    )

                # MASTER ONLY: Log to wandb
                if cfg_train.log_wandb and is_master():
                    run.log(
                        {
                            **{
                                f"training_stats/{key}": value
                                for key, value in stats.items()
                            },
                        },
                        step=global_step,
                    )

                    # Check for and log new lmms-eval results
                    eval_results_dir = os.path.join("eval_results", run_name)
                    if os.path.exists(eval_results_dir):
                        logged_results_count = 0
                        for result_file in os.listdir(eval_results_dir):
                            # Match only files like "step_1234.json" (no extra text)
                            match = re.fullmatch(r"step_(\d+)\.json", result_file)
                            if not match:
                                continue  # skip if the filename has extra text like taskname

                            try:
                                step = int(match.group(1))
                                if step not in logged_eval_steps:
                                    with open(
                                        os.path.join(eval_results_dir, result_file), "r"
                                    ) as f:
                                        eval_data = json.load(f)

                                    lmms_results = eval_data.get("results", {})
                                    if lmms_results:
                                        metrics = {
                                            f"lmms_eval/{key}": value
                                            for key, value in lmms_results.items()
                                        }
                                        metrics[lmms_eval_step] = eval_data[
                                            "global_step"
                                        ]
                                        if logged_results_count > 0:
                                            print(
                                                f"Logging more than one lmms-eval result for step {global_step}, try to avoid this."
                                            )
                                        run.log(
                                            metrics,
                                            step=global_step + logged_results_count,
                                        )
                                        logged_results_count += 1
                                        print(
                                            f"Logged lmms-eval results from step {eval_data['global_step']}"
                                        )

                                    logged_eval_steps.add(step)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                print(
                                    f"Warning: Could not process eval result file {result_file}. Error: {e}"
                                )
                                continue

                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = dist_mean_scalar(batch_loss)
                else:
                    batch_loss_gathered = batch_loss

                # MASTER ONLY: Log to wandb
                if cfg_train.log_wandb and is_master():
                    run.log(
                        {
                            "batch_loss": batch_loss_gathered,
                            **(
                                {"grad_norm": grad_norm}
                                if cfg_train.max_grad_norm is not None
                                else {}
                            ),
                        },
                        step=global_step,
                    )

            if is_update_step:
                global_step += 1
                if global_step >= cfg_train.max_training_steps:
                    break
            data_load_start = time.time()

        iter_train_loader = iter(train_loader)
        avg_train_loss = total_train_loss / i
        # gather average batch loss from all ranks if DDP
        avg_train_loss = (
            mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = (
            sum(dist_gather(total_tokens_processed))
            if is_dist()
            else total_tokens_processed
        )
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if cfg_train.log_wandb:
                run.log(
                    {
                        "epoch_loss": avg_train_loss,
                        "epoch_duration": epoch_duration,
                        "epoch_tokens_per_second": epoch_tokens_per_second,
                    }
                )

            print(
                f"Epoch: {epoch}, Step: {global_step}/{cfg_train.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}"
            )

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(
            cfg_train.batch_size
            * get_world_size()
            * cfg_train.gradient_accumulation_steps
        )
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if cfg_vlm.hf_repo_name is not None and best_model_path:
            print(
                f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub..."
            )
            hf_model = VisionLanguageModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(cfg_vlm.hf_repo_name)

        if cfg_train.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()


# %%
def main():
    global PG_CPU
    cfg_train = TrainConfig()
    cfg_vlm = VLMConfig()

    if is_master():
        print("--- Starting Training ---")
        print("--- Train Config ---")
        print(f"Dataset: {cfg_train.dataset_name}: {cfg_train.train_dataset_path}")
        print(f"Subdataset: {cfg_train.train_dataset_name}")
        print(f"Batch size: {cfg_train.batch_size}")

        if cfg_train.compile:
            print(f"Compile: {cfg_train.compile_mode}\n")
        else:
            print("Non-compiled\n")

    train(cfg_train, cfg_vlm)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()
