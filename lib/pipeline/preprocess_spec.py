"""PreprocessSpec — a resolution/normalization/device descriptor for pipeline preprocessing.

Each AI model in a pipeline declares what preprocessed tensor it needs
(resolution, normalization, device, precision).  The preprocessor collects
all unique specs and produces one tensor per spec from a single disk/video
read.  Models are wired to consume the output matching their spec's key.
"""

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn.functional as F


# Normalization presets: id → (mean, std) for per-channel [0,1]-scaled tensors.
NORMALIZATION_PRESETS = {
    0: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),             # ImageNet
    1: ([0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),                      # CLIP
}


@dataclass(frozen=True)
class PreprocessSpec:
    """Immutable descriptor for one preprocessed tensor a pipeline needs.

    The preprocessor produces one output per unique spec.  Models are wired
    to consume the output whose key matches their spec.
    """

    width: int = 0            # Target width.  0 = preserve original aspect.
    height: int = 0           # Target height. 0 = preserve original aspect.
    normalization: int = -1   # -1 = raw [0,255]; 0 = ImageNet; 1 = CLIP.
    device: str = "gpu"       # "cpu" or "gpu".
    half_precision: bool = False
    max_long_edge: int = 0    # 0 = no cap; >0 = cap longest edge (preserving aspect).

    @property
    def key(self) -> str:
        """Deterministic, human-readable pipeline-data key."""
        size = f"{self.width}x{self.height}" if (self.width or self.height) else "native"
        norm = "raw" if self.normalization == -1 else f"n{self.normalization}"
        prec = "f16" if self.half_precision else "f32"
        cap = f"_max{self.max_long_edge}" if self.max_long_edge > 0 else ""
        return f"prep__{size}_{norm}_{self.device}_{prec}{cap}"

    @property
    def effective_resolution(self) -> int:
        """Approximate pixel dimension for sorting (highest-first).

        0 means 'original' and sorts as largest.
        """
        fixed = max(self.width, self.height)
        cap = self.max_long_edge
        if fixed > 0:
            return fixed
        if cap > 0:
            return cap
        return 999_999  # native = treat as largest

    # ── Factory helpers ────────────────────────────────────────────────

    @staticmethod
    def for_model(model_inner) -> "PreprocessSpec":
        """Auto-derive a spec from a model's config properties.

        Checks for an explicit ``preprocess_config`` dict first (fully
        customizable), then falls back to role-based heuristics.
        """
        pc = getattr(model_inner, "preprocess_config", None)
        if pc and isinstance(pc, dict):
            size = pc.get("image_size", getattr(model_inner, "model_image_size", 512) or 512)
            return PreprocessSpec(
                width=pc.get("width", size),
                height=pc.get("height", size),
                normalization=pc.get("normalization", -1),
                device=pc.get("device", "gpu"),
                half_precision=pc.get("half_precision", False),
                max_long_edge=pc.get("max_long_edge", 0),
            )

        face_role = getattr(model_inner, "face_model_role", None)
        if face_role is not None:
            return PreprocessSpec.for_detector(model_inner)

        capabilities = set(getattr(model_inner, "model_capabilities", None) or [])
        if "detection" in capabilities:
            return PreprocessSpec.for_detector(model_inner)

        return PreprocessSpec.for_tagging_model(model_inner)

    @staticmethod
    def for_tagging_model(model_inner) -> "PreprocessSpec":
        """Spec for a standard image-tagging model."""
        size = getattr(model_inner, "model_image_size", 512) or 512
        norm = getattr(model_inner, "normalization_config", None)
        if norm is None or norm < 0:
            norm = 1  # default CLIP
        return PreprocessSpec(
            width=size, height=size,
            normalization=norm, device="gpu",
            half_precision=True, max_long_edge=0,
        )

    @staticmethod
    def for_detector(model_inner) -> "PreprocessSpec":
        """Spec for a detection model (face detector, etc.)."""
        size = getattr(model_inner, "model_image_size", 640) or 640
        return PreprocessSpec(
            width=0, height=0,
            normalization=-1, device="gpu",
            half_precision=False, max_long_edge=size,
        )

    @staticmethod
    def for_region_source(max_long_edge: Optional[int] = None) -> "PreprocessSpec":
        """Spec for a full-resolution region-source tensor (face/region cropping).

        Stays on CPU to avoid GPU memory pressure from large tensors.
        """
        if max_long_edge is None:
            try:
                max_long_edge = int(os.environ.get("REGION_SOURCE_MAX_LONG_EDGE", "1920"))
            except (ValueError, TypeError):
                max_long_edge = 1920
        return PreprocessSpec(
            width=0, height=0,
            normalization=-1, device="cpu",
            half_precision=False, max_long_edge=max_long_edge,
        )


# ── Tensor processing ─────────────────────────────────────────────────

def apply_spec(source: torch.Tensor, spec: PreprocessSpec) -> torch.Tensor:
    """Apply a *PreprocessSpec* to a raw ``float32`` ``[0, 255]`` CHW tensor.

    The source tensor is never modified in-place.  Returns a new tensor on
    the spec's target device with the requested resolution, normalization,
    and precision.
    """
    t = source
    h, w = t.shape[-2], t.shape[-1]
    resized = False

    # 1. Resize -----------------------------------------------------------
    if spec.width > 0 and spec.height > 0:
        if h != spec.height or w != spec.width:
            t = F.interpolate(
                t.unsqueeze(0), size=(spec.height, spec.width),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            resized = True
    elif spec.max_long_edge > 0:
        long_edge = max(h, w)
        if long_edge > spec.max_long_edge:
            scale = spec.max_long_edge / long_edge
            new_h = max(1, round(h * scale))
            new_w = max(1, round(w * scale))
            t = F.interpolate(
                t.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            resized = True

    # Ensure we never alias the source tensor.
    if not resized:
        t = t.clone()

    # 2. Device -----------------------------------------------------------
    target_device = (
        torch.device("cpu") if spec.device == "cpu"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if t.device != target_device:
        t = t.to(target_device)

    # 3. Normalize (after resize, on target device) -----------------------
    if spec.normalization >= 0:
        mean_list, std_list = NORMALIZATION_PRESETS[spec.normalization]
        t = t / 255.0
        mean = torch.tensor(mean_list, device=t.device, dtype=t.dtype).view(-1, 1, 1)
        std = torch.tensor(std_list, device=t.device, dtype=t.dtype).view(-1, 1, 1)
        t = (t - mean) / std

    # 4. Precision --------------------------------------------------------
    if spec.half_precision:
        t = t.half()

    return t
