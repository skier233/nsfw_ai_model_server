from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import warnings
from pathlib import Path

import torch


logger = logging.getLogger("logger")

# Suppress the one-time "buffer is not writable" warning emitted by
# torch.export.load → torch.frombuffer when loading .pt2/.ep archives.
warnings.filterwarnings("ignore", message=".*buffer is not writable.*", category=UserWarning)

_MODELS_DIR = Path("./models")
# torch.export.load handles both plain .pt2 programs and torch_tensorrt .ep
# programs.  Prefer the portable FP16 .pt2: the text encoder runs once per
# search query and is dominated by a memory-bound embedding gather, so a
# TensorRT engine gives no meaningful speedup while bloating the artifact (TRT
# bakes the ~900k-row embedding table into a >5 GB engine).  The .ep variants
# remain accepted as a fallback for anyone who builds one deliberately.
# Suffixes mirror what scripts/export_semtext.py writes.
def _export_candidates(model_file_name: str) -> tuple[str, ...]:
    return (f"{model_file_name}.pt2", f"{model_file_name}_trt.ep", f"{model_file_name}.ep")


@dataclass(frozen=True)
class _TextEncoderSpec:
    kind_family: str
    model_file_name: str    # exported artifact under ./models/<name>.{ep,pt2}
    tokenizer_dir_name: str  # bundled tokenizer directory under ./models/<dir>
    model_key: str
    seq_len: int = 77       # CLIP context length; must match the exported graph


def _resolve_export_path(model_file_name: str) -> Path:
    candidates = _export_candidates(model_file_name)
    for file_name in candidates:
        candidate = _MODELS_DIR / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No exported text encoder found for '{model_file_name}' "
        f"(looked for {', '.join(candidates)} in {_MODELS_DIR}). "
        f"Export it with: python scripts/export_semtext.py"
    )


class _SemanticTextEncoder:
    def __init__(self, spec: _TextEncoderSpec):
        try:
            from transformers import AutoTokenizer, logging as transformers_logging
        except ImportError as exc:  # pragma: no cover - handled at runtime by endpoint
            raise RuntimeError(
                "Text encoding requires the 'transformers', 'tokenizers', and 'safetensors' packages."
            ) from exc

        self._spec = spec
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = _resolve_export_path(spec.model_file_name)
        logger.info(
            "Loading exported text encoder for kind family '%s' from %s on %s",
            spec.kind_family,
            model_path,
            self._device,
        )

        # Tokenizer is loaded from the bundled local directory written at export
        # time — no source model identifier and no network round-trip.
        tokenizer_dir = _MODELS_DIR / spec.tokenizer_dir_name
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                f"No bundled tokenizer found at {tokenizer_dir}. "
                f"Export it with: python scripts/export_semtext.py"
            )
        previous_transformers_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        finally:
            transformers_logging.set_verbosity(previous_transformers_verbosity)

        # A TensorRT .ep program references torch.ops.tensorrt.execute_engine,
        # which is only registered once torch_tensorrt is imported.
        if model_path.suffix == ".ep":
            try:
                import torch_tensorrt  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"Loading the TensorRT text encoder '{model_path.name}' requires torch_tensorrt. "
                    f"Install it, or export a portable .pt2 with: python scripts/export_semtext.py"
                ) from exc

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*buffer is not writable.*", category=UserWarning)
            if model_path.suffix == ".ep":
                # TensorRT programs are compiled for a specific CUDA target and
                # must not be retargeted to another device.
                from torch.export import load as export_load
                self._model = export_load(str(model_path)).module().to(self._device)
            else:
                from lib.utils.torch_export_loader import load_exported_module
                self._model = load_exported_module(str(model_path), self._device)

    def encode(self, text: str) -> dict:
        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("Text is required.")

        # Pad to the exact sequence length the graph was exported with — only the
        # batch dimension is dynamic in the exported program.
        encoded = self._tokenizer(
            normalized_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._spec.seq_len,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.inference_mode():
            # The exported graph already L2-normalizes and returns float32.
            outputs = self._model(input_ids, attention_mask)
            vector = outputs[0].detach().float().cpu()

        return {
            "vector": vector.tolist(),
            "dim": int(vector.numel()),
            "model_key": self._spec.model_key,
        }


class TextEncodingService:
    _SPECS = {
        "semantic.v1": _TextEncoderSpec(
            kind_family="semantic.v1",
            model_file_name="semtext",
            tokenizer_dir_name="semtext_tokenizer",
            model_key="semvisual",
        ),
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._encoders: dict[str, _SemanticTextEncoder] = {}

    def encode(self, kind_family: str, text: str) -> dict:
        normalized_kind_family = (kind_family or "").strip().lower()
        if not normalized_kind_family:
            raise ValueError("kind_family is required.")

        encoder = self._get_encoder(normalized_kind_family)
        return encoder.encode(text)

    def _get_encoder(self, kind_family: str) -> _SemanticTextEncoder:
        with self._lock:
            encoder = self._encoders.get(kind_family)
            if encoder is not None:
                return encoder

            spec = self._SPECS.get(kind_family)
            if spec is None:
                raise ValueError(f"No text encoder is configured for kind family '{kind_family}'.")

            encoder = _SemanticTextEncoder(spec)
            self._encoders[kind_family] = encoder
            return encoder


text_encoding_service = TextEncodingService()
