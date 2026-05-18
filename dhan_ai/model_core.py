"""
model_core.py — LLM Model Architecture, Fine-Tuning & Inference

Core module for building, fine-tuning, and running inference on
large language models tailored to Indian financial market intelligence.

Supports:
  - HuggingFace transformer model loading
  - LoRA / QLoRA parameter-efficient fine-tuning (PEFT)
  - 4-bit and 8-bit quantization via bitsandbytes
  - Financial-domain tokenizer extension
  - Adapter merging and export for deployment
  - Streaming and batched inference
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class ModelTask(str, Enum):
    """Supported model tasks."""

    CAUSAL_LM = "causal_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"


class QuantizationMode(str, Enum):
    """Quantization precision levels."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


FINANCIAL_SPECIAL_TOKENS = [
    "<|market_open|>",
    "<|market_close|>",
    "<|bullish|>",
    "<|bearish|>",
    "<|neutral|>",
    "<|nifty|>",
    "<|sensex|>",
    "<|buy_signal|>",
    "<|sell_signal|>",
    "<|hold_signal|>",
    "<|earnings|>",
    "<|dividend|>",
    "<|ipo|>",
    "<|sector|>",
    "<|analysis_start|>",
    "<|analysis_end|>",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for model loading, quantization, and LoRA fine-tuning.

    Attributes:
        model_name_or_path: HuggingFace model ID or local path.
        task: Model task type (causal LM or sequence classification).
        num_labels: Number of labels for classification tasks.
        quantization: Quantization mode (none / int8 / int4).
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization.
        bnb_4bit_quant_type: Quantization type for 4-bit (nf4 / fp4).
        use_double_quant: Enable nested quantization for QLoRA.
        use_lora: Whether to apply LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout in LoRA layers.
        lora_target_modules: Module names to apply LoRA to.
        lora_bias: LoRA bias handling strategy.
        add_financial_tokens: Whether to extend tokenizer with
            domain-specific financial tokens.
        max_seq_length: Maximum sequence length.
        torch_dtype: Model weight dtype.
        device_map: Device placement strategy.
        trust_remote_code: Trust remote model code from HuggingFace.
        attn_implementation: Attention implementation (eager / sdpa / flash_attention_2).
    """

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    task: ModelTask = ModelTask.CAUSAL_LM
    num_labels: int = 3

    # Quantization
    quantization: QuantizationMode = QuantizationMode.NONE
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # LoRA / PEFT
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_bias: str = "none"

    # Tokenizer
    add_financial_tokens: bool = True
    max_seq_length: int = 2048

    # Loading
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None

    def resolve_torch_dtype(self) -> torch.dtype:
        """Resolve the string dtype to a torch.dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-compatible dict."""
        data: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                data[k] = v.value
            else:
                data[k] = v
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Deserialize config from a dict."""
        if "task" in data:
            data["task"] = ModelTask(data["task"])
        if "quantization" in data:
            data["quantization"] = QuantizationMode(data["quantization"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Union[str, Path]) -> None:
        """Save config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class StopOnTokens(StoppingCriteria):
    """Stop generation when any of the specified token sequences appear."""

    def __init__(self, stop_token_ids: List[List[int]]) -> None:
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            seq_len = len(stop_ids)
            if input_ids.shape[-1] >= seq_len:
                if input_ids[0, -seq_len:].tolist() == stop_ids:
                    return True
        return False


# ---------------------------------------------------------------------------
# Core Model Class
# ---------------------------------------------------------------------------

class FinancialLLM:
    """High-level wrapper for loading, fine-tuning, and running inference
    on a large language model for Indian financial market intelligence.

    Example::

        config = ModelConfig(
            model_name_or_path="meta-llama/Llama-2-7b-hf",
            quantization=QuantizationMode.INT4,
            use_lora=True,
        )
        model = FinancialLLM(config)
        model.load()

        # Fine-tuning handled by TrainingOrchestrator
        # Inference:
        output = model.generate("Analyze NIFTY 50 trend for Q1 2025")
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._is_loaded = False
        self._adapter_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "FinancialLLM":
        """Load the model and tokenizer from the configured source."""
        logger.info("Loading tokenizer from %s", self.config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            model_max_length=self.config.max_seq_length,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.config.add_financial_tokens:
            self._extend_tokenizer()

        quant_config = self._build_quantization_config()

        model_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.model_name_or_path,
            "torch_dtype": self.config.resolve_torch_dtype(),
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        logger.info("Loading model: %s (task=%s)", self.config.model_name_or_path, self.config.task.value)

        if self.config.task == ModelTask.CAUSAL_LM:
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        elif self.config.task == ModelTask.SEQUENCE_CLASSIFICATION:
            model_kwargs["num_labels"] = self.config.num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)

        if self.config.add_financial_tokens and self.model is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.config.use_lora:
            self._apply_lora()

        self._is_loaded = True
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Model loaded — total params: %s, trainable: %s (%.2f%%)",
            f"{param_count:,}",
            f"{trainable:,}",
            100.0 * trainable / max(param_count, 1),
        )
        return self

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Build the bitsandbytes quantization config if requested."""
        if self.config.quantization == QuantizationMode.NONE:
            return None

        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(
            self.config.bnb_4bit_compute_dtype, torch.bfloat16
        )

        if self.config.quantization == QuantizationMode.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
        )

    def _extend_tokenizer(self) -> None:
        """Add financial domain-specific tokens to the tokenizer."""
        new_tokens = [
            tok for tok in FINANCIAL_SPECIAL_TOKENS
            if tok not in self.tokenizer.get_vocab()
        ]
        if new_tokens:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            logger.info("Added %d financial special tokens to tokenizer", num_added)

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model using PEFT."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT is required for LoRA fine-tuning. "
                "Install it with: pip install peft"
            )

        if self.config.quantization != QuantizationMode.NONE:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        task_type = (
            TaskType.CAUSAL_LM
            if self.config.task == ModelTask.CAUSAL_LM
            else TaskType.SEQ_CLS
        )
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias,
            task_type=task_type,
        )
        self.model = get_peft_model(self.model, lora_config)
        self._adapter_name = "default"
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def load_adapter(self, adapter_path: Union[str, Path], adapter_name: str = "financial") -> None:
        """Load a pre-trained LoRA adapter from disk."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required to load adapters.")
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), adapter_name=adapter_name
            )
        else:
            self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)
        self._adapter_name = adapter_name
        logger.info("Loaded adapter '%s' from %s", adapter_name, adapter_path)

    def merge_and_unload(self) -> PreTrainedModel:
        """Merge LoRA weights into the base model and unload PEFT wrapper.

        Returns the merged base model ready for deployment.
        """
        if not PEFT_AVAILABLE or not isinstance(self.model, PeftModel):
            logger.warning("No PEFT model to merge; returning model as-is")
            return self.model
        logger.info("Merging adapter weights into base model")
        self.model = self.model.merge_and_unload()
        self._adapter_name = None
        return self.model

    # ------------------------------------------------------------------
    # Saving & export
    # ------------------------------------------------------------------

    def save_pretrained(self, output_dir: Union[str, Path], save_merged: bool = False) -> Path:
        """Save the model and tokenizer.

        Args:
            output_dir: Directory to write model artifacts.
            save_merged: If True, merge adapters before saving.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_merged and PEFT_AVAILABLE and isinstance(self.model, PeftModel):
            merged = self.model.merge_and_unload()
            merged.save_pretrained(str(output_dir))
        else:
            self.model.save_pretrained(str(output_dir))

        self.tokenizer.save_pretrained(str(output_dir))
        self.config.save(output_dir / "dhan_model_config.json")
        logger.info("Model saved to %s", output_dir)
        return output_dir

    def push_to_hub(self, repo_id: str, private: bool = True) -> None:
        """Push model and tokenizer to HuggingFace Hub."""
        self.model.push_to_hub(repo_id, private=private)
        self.tokenizer.push_to_hub(repo_id, private=private)
        logger.info("Pushed model to HuggingFace Hub: %s", repo_id)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty factor.
            do_sample: Whether to sample or use greedy decoding.
            stop_strings: Optional stop strings to halt generation.

        Returns:
            Generated text (excluding the prompt).
        """
        self._ensure_loaded()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        stopping_criteria = None
        if stop_strings:
            stop_ids = [
                self.tokenizer.encode(s, add_special_tokens=False)
                for s in stop_strings
            ]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

        with autocast(dtype=self.config.resolve_torch_dtype()):
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Stream generated tokens one at a time.

        Yields:
            Individual decoded tokens as strings.
        """
        self._ensure_loaded()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        import threading

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={**inputs, "generation_config": gen_config, "streamer": streamer},
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    @torch.inference_mode()
    def predict_sentiment(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run sentiment/classification inference on financial texts.

        Args:
            texts: Single text or list of texts.
            batch_size: Batch size for inference.

        Returns:
            List of dicts with 'label', 'score', and 'logits' keys.
        """
        self._ensure_loaded()
        if self.config.task != ModelTask.SEQUENCE_CLASSIFICATION:
            raise ValueError(
                "predict_sentiment requires task=SEQUENCE_CLASSIFICATION"
            )

        if isinstance(texts, str):
            texts = [texts]

        label_map = {0: "bearish", 1: "neutral", 2: "bullish"}
        results: List[Dict[str, Any]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.model.device)

            with autocast(dtype=self.config.resolve_torch_dtype()):
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            pred_ids = probs.argmax(dim=-1)

            for j in range(len(batch)):
                pred_id = pred_ids[j].item()
                results.append({
                    "label": label_map.get(pred_id, str(pred_id)),
                    "score": probs[j, pred_id].item(),
                    "logits": outputs.logits[j].cpu().tolist(),
                })

        return results

    @torch.inference_mode()
    def encode(
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> torch.Tensor:
        """Compute hidden-state embeddings for texts.

        Useful for similarity search and retrieval in financial data.

        Args:
            texts: Input text(s).
            normalize: L2-normalize the embeddings.

        Returns:
            Tensor of shape (N, hidden_size).
        """
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        with autocast(dtype=self.config.resolve_torch_dtype()):
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_model(self) -> PreTrainedModel:
        """Return the underlying model (for use with TrainingOrchestrator)."""
        self._ensure_loaded()
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer."""
        self._ensure_loaded()
        return self.tokenizer

    def trainable_parameter_summary(self) -> Dict[str, Any]:
        """Return a summary of parameter counts."""
        self._ensure_loaded()
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percent": round(100.0 * trainable / max(total, 1), 4),
        }

    def _ensure_loaded(self) -> None:
        """Raise if the model has not been loaded."""
        if not self._is_loaded or self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call .load() before inference."
            )

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"FinancialLLM(model={self.config.model_name_or_path!r}, "
            f"task={self.config.task.value}, status={status})"
        )
