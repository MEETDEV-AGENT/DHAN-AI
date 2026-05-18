"""
training_orchestrator.py — Distributed Training Orchestration

End-to-end orchestrator for fine-tuning LLMs on financial market data.

Supports:
  - Single-GPU and multi-GPU training (PyTorch DDP / FSDP)
  - Mixed-precision training (fp16 / bf16) with gradient scaling
  - Gradient accumulation for effective large batch sizes
  - Checkpoint management (save / resume / best-model tracking)
  - Learning rate scheduling (warmup + cosine / linear decay)
  - Early stopping with configurable patience
  - W&B and TensorBoard logging
  - Financial dataset preprocessing pipeline
  - Evaluation loops with market-specific metrics
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from dhan_ai.model_core import FinancialLLM, ModelConfig, ModelTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DistributedBackend(str, Enum):
    """Distributed training backend."""

    NONE = "none"
    DDP = "ddp"
    FSDP = "fsdp"


class SchedulerType(str, Enum):
    """Learning rate scheduler type."""

    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"


class LoggingBackend(str, Enum):
    """Metrics logging backend."""

    CONSOLE = "console"
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Full training configuration.

    Attributes:
        output_dir: Directory for checkpoints, logs, and final model.
        run_name: Human-readable run name for logging.
        epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per GPU.
        per_device_eval_batch_size: Evaluation batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate gradients.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        adam_beta1: AdamW beta1.
        adam_beta2: AdamW beta2.
        adam_epsilon: AdamW epsilon.
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_ratio: Fraction of total steps for LR warmup.
        warmup_steps: Explicit warmup steps (overrides warmup_ratio).
        scheduler_type: LR scheduler type.
        fp16: Use fp16 mixed precision.
        bf16: Use bf16 mixed precision.
        distributed_backend: Distributed strategy.
        fsdp_sharding_strategy: FSDP sharding strategy name.
        gradient_checkpointing: Enable gradient checkpointing.
        logging_steps: Log metrics every N steps.
        eval_steps: Evaluate every N steps (0 = end of epoch only).
        save_steps: Save checkpoint every N steps (0 = end of epoch only).
        save_total_limit: Maximum checkpoints to keep.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_threshold: Minimum improvement to reset patience.
        logging_backend: Where to log metrics.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team.
        seed: Random seed.
        dataloader_num_workers: DataLoader worker count.
        pin_memory: Pin memory in DataLoader.
        resume_from_checkpoint: Path to checkpoint to resume from.
    """

    output_dir: str = "./output/dhan_training"
    run_name: str = "dhan-financial-llm"

    # Training loop
    epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = -1

    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    scheduler_type: SchedulerType = SchedulerType.COSINE

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Distributed
    distributed_backend: DistributedBackend = DistributedBackend.NONE
    fsdp_sharding_strategy: str = "FULL_SHARD"

    # Memory optimization
    gradient_checkpointing: bool = True

    # Logging & saving
    logging_steps: int = 10
    eval_steps: int = 0
    save_steps: int = 0
    save_total_limit: int = 3
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4

    # Logging backend
    logging_backend: LoggingBackend = LoggingBackend.CONSOLE
    wandb_project: str = "dhan-ai"
    wandb_entity: Optional[str] = None

    # Reproducibility
    seed: int = 42

    # Data loading
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    # Checkpoint resume
    resume_from_checkpoint: Optional[str] = None

    @property
    def effective_batch_size(self) -> int:
        """Compute the effective global batch size."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * world_size
        )

    def save(self, path: Union[str, Path]) -> None:
        """Serialize training config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                data[k] = v.value
            else:
                data[k] = v
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Deserialize training config from JSON."""
        with open(path) as f:
            data = json.load(f)
        if "scheduler_type" in data:
            data["scheduler_type"] = SchedulerType(data["scheduler_type"])
        if "distributed_backend" in data:
            data["distributed_backend"] = DistributedBackend(data["distributed_backend"])
        if "logging_backend" in data:
            data["logging_backend"] = LoggingBackend(data["logging_backend"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Financial Data Pipeline
# ---------------------------------------------------------------------------


class FinancialTextDataset(Dataset):
    """Dataset for financial text data (news, reports, analyst notes).

    Supports two formats:
      - JSONL files with 'text' field (for causal LM)
      - JSONL files with 'text' and 'label' fields (for classification)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        task: ModelTask = ModelTask.CAUSAL_LM,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.examples: List[Dict[str, Any]] = []

        file_path = Path(file_path)
        if file_path.suffix == ".jsonl":
            self._load_jsonl(file_path)
        elif file_path.suffix == ".json":
            self._load_json(file_path)
        elif file_path.is_dir():
            self._load_directory(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info("Loaded %d examples from %s", len(self.examples), file_path)

    def _load_jsonl(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def _load_json(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            self.examples = data
        else:
            raise ValueError("JSON file must contain a list of examples")

    def _load_directory(self, path: Path) -> None:
        for file in sorted(path.glob("*.jsonl")):
            self._load_jsonl(file)
        for file in sorted(path.glob("*.json")):
            try:
                self._load_json(file)
            except ValueError:
                continue

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = example["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.task == ModelTask.CAUSAL_LM:
            item["labels"] = item["input_ids"].clone()
        elif self.task == ModelTask.SEQUENCE_CLASSIFICATION:
            item["labels"] = torch.tensor(example.get("label", 0), dtype=torch.long)

        return item


# ---------------------------------------------------------------------------
# Metrics Tracker
# ---------------------------------------------------------------------------


class MetricsTracker:
    """Tracks and logs training metrics across steps and epochs."""

    def __init__(
        self,
        config: TrainingConfig,
        total_steps: int,
    ) -> None:
        self.config = config
        self.total_steps = total_steps
        self.step_metrics: List[Dict[str, float]] = []
        self.epoch_metrics: List[Dict[str, float]] = []
        self._wandb_run = None
        self._tb_writer = None

        if config.logging_backend == LoggingBackend.WANDB and WANDB_AVAILABLE:
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                config=config.__dict__,
            )
        elif config.logging_backend == LoggingBackend.TENSORBOARD and TENSORBOARD_AVAILABLE:
            log_dir = Path(config.output_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(log_dir))

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a single training step."""
        metrics["step"] = step
        metrics["progress"] = step / max(self.total_steps, 1)
        self.step_metrics.append(metrics)

        if self.config.logging_backend == LoggingBackend.CONSOLE:
            if step % self.config.logging_steps == 0:
                parts = [f"step={step}/{self.total_steps}"]
                for k, v in metrics.items():
                    if k not in ("step", "progress"):
                        parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
                logger.info(" | ".join(parts))

        elif self._wandb_run is not None:
            wandb.log(metrics, step=step)

        elif self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(f"train/{k}", v, step)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a completed epoch."""
        metrics["epoch"] = epoch
        self.epoch_metrics.append(metrics)
        logger.info("Epoch %d: %s", epoch, json.dumps(metrics, indent=2, default=str))

        if self._wandb_run is not None:
            wandb.log({f"epoch/{k}": v for k, v in metrics.items()})
        elif self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(f"epoch/{k}", v, epoch)

    def finalize(self) -> None:
        """Flush and close logging backends."""
        if self._wandb_run is not None:
            wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def get_history(self) -> Dict[str, List[Dict[str, float]]]:
        """Return all tracked metrics."""
        return {"steps": self.step_metrics, "epochs": self.epoch_metrics}


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages saving, loading, and pruning of training checkpoints."""

    def __init__(self, output_dir: Union[str, Path], save_total_limit: int = 3) -> None:
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.best_metric: Optional[float] = None
        self.best_checkpoint: Optional[Path] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        scaler: Optional[GradScaler],
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Save a training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self._unwrap_model(model)

        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(str(checkpoint_dir))
        else:
            torch.save(unwrapped.state_dict(), checkpoint_dir / "model.pt")

        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
        }
        if scaler is not None:
            training_state["scaler_state_dict"] = scaler.state_dict()

        torch.save(training_state, checkpoint_dir / "training_state.pt")
        logger.info("Saved checkpoint at step %d to %s", global_step, checkpoint_dir)

        if is_best:
            self.best_checkpoint = checkpoint_dir
            self.best_metric = metrics.get("eval_loss", metrics.get("loss"))
            best_link = self.output_dir / "best"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_dir.name)
            logger.info("New best checkpoint: %s", checkpoint_dir)

        self._prune_old_checkpoints()
        return checkpoint_dir

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
    ) -> Dict[str, Any]:
        """Load a training checkpoint and return the training state."""
        checkpoint_path = Path(checkpoint_path)
        state_path = checkpoint_path / "training_state.pt"

        if not state_path.exists():
            raise FileNotFoundError(f"No training state found at {state_path}")

        training_state = torch.load(state_path, map_location="cpu", weights_only=False)

        unwrapped = self._unwrap_model(model)
        if hasattr(unwrapped, "from_pretrained"):
            pass  # Model already loaded via from_pretrained
        else:
            model_path = checkpoint_path / "model.pt"
            if model_path.exists():
                unwrapped.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))

        if optimizer is not None and "optimizer_state_dict" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in training_state:
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in training_state:
            scaler.load_state_dict(training_state["scaler_state_dict"])

        self.best_metric = training_state.get("best_metric")
        logger.info(
            "Resumed from checkpoint: epoch=%d, step=%d",
            training_state["epoch"],
            training_state["global_step"],
        )
        return training_state

    def _prune_old_checkpoints(self) -> None:
        """Remove oldest checkpoints exceeding the save limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        while len(checkpoints) > self.save_total_limit:
            old = checkpoints.pop(0)
            if old == self.best_checkpoint:
                continue
            shutil.rmtree(old)
            logger.info("Pruned old checkpoint: %s", old)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP to get the base model."""
        if isinstance(model, (DDP,)):
            return model.module
        if FSDP_AVAILABLE and isinstance(model, FSDP):
            return model.module
        return model


# ---------------------------------------------------------------------------
# Distributed Training Orchestrator
# ---------------------------------------------------------------------------


class TrainingOrchestrator:
    """Orchestrates end-to-end distributed training for financial LLMs.

    Example::

        model_config = ModelConfig(
            model_name_or_path="meta-llama/Llama-2-7b-hf",
            quantization=QuantizationMode.INT4,
            use_lora=True,
        )
        training_config = TrainingConfig(
            output_dir="./output/nifty_finetune",
            epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            bf16=True,
        )

        llm = FinancialLLM(model_config)
        llm.load()

        orchestrator = TrainingOrchestrator(
            config=training_config,
            financial_llm=llm,
        )
        orchestrator.train(
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
    """

    def __init__(
        self,
        config: TrainingConfig,
        financial_llm: FinancialLLM,
    ) -> None:
        self.config = config
        self.financial_llm = financial_llm
        self.model = financial_llm.get_model()
        self.tokenizer = financial_llm.get_tokenizer()

        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.scaler: Optional[GradScaler] = None
        self.checkpoint_mgr = CheckpointManager(
            config.output_dir, config.save_total_limit
        )
        self.metrics_tracker: Optional[MetricsTracker] = None

        self.global_step = 0
        self.start_epoch = 0
        self._best_eval_loss = float("inf")
        self._patience_counter = 0
        self._stop_training = False

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main_process = self.local_rank == 0

        self._setup_seed()
        self._setup_output_dir()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _setup_output_dir(self) -> None:
        """Create the output directory structure."""
        if self.is_main_process:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_distributed(self) -> None:
        """Initialize the distributed process group."""
        if self.config.distributed_backend == DistributedBackend.NONE:
            if torch.cuda.is_available():
                self.model = self.model.to(f"cuda:{self.local_rank}")
            return

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_main_process = self.local_rank == 0

        torch.cuda.set_device(self.local_rank)
        self.model = self.model.to(f"cuda:{self.local_rank}")

        if self.config.distributed_backend == DistributedBackend.DDP:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            logger.info("Wrapped model with DDP (rank=%d)", self.local_rank)

        elif self.config.distributed_backend == DistributedBackend.FSDP:
            if not FSDP_AVAILABLE:
                raise ImportError("FSDP requires PyTorch >= 2.0")

            sharding_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
            }
            sharding = sharding_map.get(
                self.config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD
            )

            mixed_precision = None
            if self.config.bf16:
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            elif self.config.fp16:
                mixed_precision = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )

            self.model = FSDP(
                self.model,
                sharding_strategy=sharding,
                mixed_precision=mixed_precision,
                device_id=torch.cuda.current_device(),
                use_orig_params=True,
            )
            logger.info(
                "Wrapped model with FSDP (rank=%d, strategy=%s)",
                self.local_rank,
                self.config.fsdp_sharding_strategy,
            )

    def _setup_optimizer(self) -> None:
        """Create the AdamW optimizer with weight decay handling."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ("bias", "LayerNorm", "layernorm", "layer_norm", "ln_")):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _setup_scheduler(self, num_training_steps: int) -> None:
        """Create the learning rate scheduler."""
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == SchedulerType.COSINE:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.scheduler_type == SchedulerType.LINEAR:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.scheduler_type == SchedulerType.CONSTANT:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)

    def _setup_scaler(self) -> None:
        """Create gradient scaler for fp16 training."""
        if self.config.fp16 and not self.config.bf16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _build_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Build a DataLoader with optional distributed sampling."""
        sampler = None
        if dist.is_initialized() and self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=shuffle,
            )
            shuffle = False

        collate_fn = None
        if self.financial_llm.config.task == ModelTask.CAUSAL_LM:
            collate_fn = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
        else:
            collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            compute_metrics: Optional function to compute custom metrics.

        Returns:
            Dict with training history and final metrics.
        """
        self._setup_distributed()

        if self.config.gradient_checkpointing:
            unwrapped = CheckpointManager._unwrap_model(self.model)
            if hasattr(unwrapped, "gradient_checkpointing_enable"):
                unwrapped.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        train_loader = self._build_dataloader(
            train_dataset,
            self.config.per_device_train_batch_size,
            shuffle=True,
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = self._build_dataloader(
                eval_dataset,
                self.config.per_device_eval_batch_size,
                shuffle=False,
            )

        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        if self.config.max_steps > 0:
            num_training_steps = self.config.max_steps
            effective_epochs = math.ceil(num_training_steps / steps_per_epoch)
        else:
            num_training_steps = steps_per_epoch * self.config.epochs
            effective_epochs = self.config.epochs

        self._setup_optimizer()
        self._setup_scheduler(num_training_steps)
        self._setup_scaler()

        self.metrics_tracker = MetricsTracker(self.config, num_training_steps)

        if self.config.resume_from_checkpoint:
            state = self.checkpoint_mgr.load(
                self.config.resume_from_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
            )
            self.global_step = state["global_step"]
            self.start_epoch = state["epoch"]
            self._best_eval_loss = state.get("best_metric", float("inf")) or float("inf")

        if self.is_main_process:
            self.config.save(Path(self.config.output_dir) / "training_config.json")
            self.financial_llm.config.save(
                Path(self.config.output_dir) / "model_config.json"
            )
            logger.info(
                "Starting training: epochs=%d, steps=%d, batch=%d, lr=%.2e",
                effective_epochs,
                num_training_steps,
                self.config.effective_batch_size,
                self.config.learning_rate,
            )

        training_start = time.time()

        for epoch in range(self.start_epoch, effective_epochs):
            if self._stop_training:
                break

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = self._train_epoch(
                train_loader, epoch, num_training_steps
            )

            epoch_metrics: Dict[str, Any] = {"train_loss": epoch_loss}

            if eval_loader is not None and self.is_main_process:
                eval_metrics = self._evaluate(eval_loader, compute_metrics)
                epoch_metrics.update(eval_metrics)

                eval_loss = eval_metrics.get("eval_loss", float("inf"))
                is_best = eval_loss < self._best_eval_loss - self.config.early_stopping_threshold
                if is_best:
                    self._best_eval_loss = eval_loss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

                if self._patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement",
                        self._patience_counter,
                    )
                    self._stop_training = True
            else:
                is_best = False

            if self.is_main_process:
                self.metrics_tracker.log_epoch(epoch, epoch_metrics)

                if self.config.save_steps == 0:
                    self.checkpoint_mgr.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        epoch,
                        self.global_step,
                        epoch_metrics,
                        is_best=is_best,
                    )

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        training_time = time.time() - training_start

        if self.is_main_process:
            self._save_final_model()
            self.metrics_tracker.finalize()
            logger.info(
                "Training complete: %.2f minutes, %d steps, best_eval_loss=%.6f",
                training_time / 60,
                self.global_step,
                self._best_eval_loss,
            )

        if dist.is_initialized():
            dist.destroy_process_group()

        return {
            "training_time_seconds": training_time,
            "global_step": self.global_step,
            "best_eval_loss": self._best_eval_loss,
            "history": self.metrics_tracker.get_history() if self.metrics_tracker else {},
        }

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_steps: int,
    ) -> float:
        """Run a single training epoch. Returns the average epoch loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_loss = 0.0

        mixed_precision_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        for step, batch in enumerate(train_loader):
            if self._stop_training:
                break
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

            batch = {k: v.to(f"cuda:{self.local_rank}") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(
                dtype=mixed_precision_dtype,
                enabled=(self.config.fp16 or self.config.bf16),
            ):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss += loss.item()
            num_batches += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulation_loss

                if self.is_main_process and self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.metrics_tracker.log_step(
                        self.global_step,
                        {
                            "loss": accumulation_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch,
                            "gpu_memory_mb": (
                                torch.cuda.max_memory_allocated() / 1024 / 1024
                                if torch.cuda.is_available()
                                else 0
                            ),
                        },
                    )

                if (
                    self.config.save_steps > 0
                    and self.global_step % self.config.save_steps == 0
                    and self.is_main_process
                ):
                    self.checkpoint_mgr.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        epoch,
                        self.global_step,
                        {"loss": accumulation_loss},
                    )

                accumulation_loss = 0.0

        steps_completed = max(
            num_batches // self.config.gradient_accumulation_steps, 1
        )
        return total_loss / steps_completed

    @torch.inference_mode()
    def _evaluate(
        self,
        eval_loader: DataLoader,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Run evaluation loop and return metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        mixed_precision_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        for batch in eval_loader:
            batch = {k: v.to(f"cuda:{self.local_rank}") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(
                dtype=mixed_precision_dtype,
                enabled=(self.config.fp16 or self.config.bf16),
            ):
                outputs = self.model(**batch)

            total_loss += outputs.loss.item()
            num_batches += 1

            if compute_metrics is not None and hasattr(outputs, "logits"):
                preds = outputs.logits.argmax(dim=-1).cpu()
                labels = batch["labels"].cpu()
                all_preds.append(preds)
                all_labels.append(labels)

        metrics: Dict[str, float] = {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_perplexity": math.exp(
                min(total_loss / max(num_batches, 1), 100)
            ),
        }

        if compute_metrics is not None and all_preds:
            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)
            custom = compute_metrics(preds_cat, labels_cat)
            metrics.update(custom)

        self.model.train()
        return metrics

    def _save_final_model(self) -> None:
        """Save the final trained model."""
        final_dir = Path(self.config.output_dir) / "final_model"
        self.financial_llm.save_pretrained(final_dir, save_merged=False)
        logger.info("Final model saved to %s", final_dir)

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> "TrainingOrchestrator":
        """Create a TrainingOrchestrator from model and training configs.

        Loads the model and returns a ready-to-train orchestrator.
        """
        llm = FinancialLLM(model_config)
        llm.load()
        return cls(config=training_config, financial_llm=llm)

    @classmethod
    def launch_distributed(
        cls,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_data_path: Union[str, Path],
        eval_data_path: Optional[Union[str, Path]] = None,
        num_gpus: Optional[int] = None,
    ) -> None:
        """Launch distributed training using torchrun.

        This is a convenience method that builds the launch command
        for multi-GPU training.

        Args:
            model_config: Model configuration.
            training_config: Training configuration.
            train_data_path: Path to training data.
            eval_data_path: Path to evaluation data.
            num_gpus: Number of GPUs (defaults to all available).
        """
        import subprocess
        import sys

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        model_config.save("/tmp/dhan_model_config.json")
        training_config.save("/tmp/dhan_training_config.json")

        launch_script = Path(__file__).parent / "_distributed_launcher.py"

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            str(launch_script),
            "--model_config",
            "/tmp/dhan_model_config.json",
            "--training_config",
            "/tmp/dhan_training_config.json",
            "--train_data",
            str(train_data_path),
        ]
        if eval_data_path:
            cmd.extend(["--eval_data", str(eval_data_path)])

        logger.info("Launching distributed training: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
