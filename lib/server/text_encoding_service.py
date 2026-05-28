from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

import torch
import torch.nn.functional as F


logger = logging.getLogger("logger")


@dataclass(frozen=True)
class _TextEncoderSpec:
    kind_family: str
    model_source: str
    model_key: str


class _SemanticTextEncoder:
    def __init__(self, spec: _TextEncoderSpec):
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection, logging as transformers_logging
        except ImportError as exc:  # pragma: no cover - handled at runtime by endpoint
            raise RuntimeError(
                "Text encoding requires the 'transformers', 'tokenizers', and 'safetensors' packages."
            ) from exc

        self._spec = spec
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {}
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        logger.info(
            "Loading text encoder for kind family '%s' from %s on %s",
            spec.kind_family,
            spec.model_key,
            self._device,
        )

        previous_transformers_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.model_source)
            self._model = CLIPTextModelWithProjection.from_pretrained(spec.model_source, **model_kwargs)
        finally:
            transformers_logging.set_verbosity(previous_transformers_verbosity)
        self._model.eval()
        self._model.to(self._device)

    def encode(self, text: str) -> dict:
        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("Text is required.")

        encoded = self._tokenizer(
            normalized_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            outputs = self._model(**encoded)
            vector = F.normalize(outputs.text_embeds.float(), dim=-1)[0].detach().cpu()

        return {
            "vector": vector.tolist(),
            "dim": int(vector.numel()),
            "model_key": self._spec.model_key,
        }


class TextEncodingService:
    _SEMANTIC_TEXT_MODEL_SOURCE = "facebook/" + "meta" + "clip-2-worldwide-huge-quickgelu"
    _SPECS = {
        "semantic.v1": _TextEncoderSpec(
            kind_family="semantic.v1",
            model_source=_SEMANTIC_TEXT_MODEL_SOURCE,
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