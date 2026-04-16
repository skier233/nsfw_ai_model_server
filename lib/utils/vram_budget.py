import logging
import os
import torch

logger = logging.getLogger("logger")

# Overhead for CUDA context, PyTorch caching allocator batch buffers,
# display driver, and uncalibrated model weight underestimates.
# Validated against multi-model nvidia-smi calibration (8 models, RTX 3090):
#   overhead=2000 + util=0.9 → batch=26 for proud_totem, ~1241 MB headroom.
_DEFAULT_OVERHEAD_MB = 2000

# Safety factor to account for PyTorch caching allocator fragmentation,
# cuDNN workspace retention, and estimation inaccuracies.
# 0.9 = target 90% utilization of estimated available VRAM.
_UTILIZATION_FACTOR = 0.9


def get_total_vram_mb(device_str):
    """Return total VRAM in MB for the given device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    if device_str == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_properties(0).total_memory / (1024 ** 2)
    return 0


def estimate_model_weight_mb(model_file_name, model_license_name=None):
    """Estimate on-GPU weight memory from the model file size on disk."""
    if model_license_name is not None:
        if model_license_name.endswith(".0"):
            path = f"./models/{model_file_name}.pt.enc"
        else:
            path = f"./models/{model_file_name}.pt2.enc"
    else:
        path = None
        for ext in (".pt2", ".pt"):
            candidate = f"./models/{model_file_name}{ext}"
            if os.path.exists(candidate):
                path = candidate
                break

    if path and os.path.exists(path):
        return os.path.getsize(path) / (1024 ** 2)
    return 200.0  # Conservative fallback when file is missing


def _estimate_preprocess_tensor_mb(image_size, half_precision=True):
    """Estimate GPU memory for one preprocessed tensor (CHW) at given size."""
    bytes_per_element = 2 if half_precision else 4
    return (3 * image_size * image_size * bytes_per_element) / (1024 ** 2)


def _collect_gpu_preprocess_specs(ai_models):
    """Derive unique GPU preprocess specs from all AI models.

    Returns list of (width, half_precision) tuples for specs
    that will reside on GPU.
    """
    seen = set()
    specs = []
    for model in ai_models:
        image_size = getattr(model, 'model_image_size', None)
        if image_size is None or image_size <= 0:
            continue
        device = getattr(model, 'device', 'cuda')
        if device in ('cpu', 'mps'):
            continue
        capabilities = set(getattr(model, 'model_capabilities', []) or [])
        half = 'detection' not in capabilities
        key = (image_size, half)
        if key not in seen:
            seen.add(key)
            specs.append((image_size, half))
    return specs


def _round_to_8(n):
    """Round batch size to nearest multiple of 8, ties round down.

    Values below 8 are returned as-is (minimum 1).
    """
    if n < 8:
        return max(1, n)
    lower = (n // 8) * 8
    upper = lower + 8
    # Round to nearest; ties (remainder == 4) go down
    if (upper - n) < (n - lower):
        return upper
    return lower


def compute_batch_sizes(ai_models, device_str, max_pending_frames=0,
                        overhead_mb=_DEFAULT_OVERHEAD_MB):
    """Compute batch sizes for all AI models using VRAM budget allocation.

    All dynamically-scaled models share the same pipeline, so the effective
    batch size is the minimum across all of them (the bottleneck model,
    typically the heaviest per-item activation like proud_totem at 512px).
    Using a uniform pipeline batch prevents fill_to_batch waste: without
    equalization, a model with batch=50 would pad incoming batches of 24
    with 26 zero tensors, wasting significant VRAM.

    Batch sizes are rounded to the nearest multiple of 8 for GPU efficiency.

    Models without dynamic scaling (e.g. face models with a fixed
    max_batch_size) are left unchanged but still counted in the weight budget.

    Args:
        ai_models: List of AIModel instances.
        device_str: Device string ("cuda", "xpu", etc.).
        max_pending_frames: Max queued video frames; 0 = images only (1 set).
        overhead_mb: Reserved VRAM for CUDA context and OS.
    """
    total_vram_mb = get_total_vram_mb(device_str)
    if total_vram_mb == 0:
        return  # CPU/MPS — no VRAM to budget

    # ── 1. Model weights (always resident) ──────────────────────────────
    weight_estimates = {}
    for model in ai_models:
        override = getattr(model, '_model_weight_mb_override', None)
        if override is not None:
            weight_estimates[id(model)] = float(override)
        else:
            weight_estimates[id(model)] = estimate_model_weight_mb(
                model.model_file_name,
                getattr(model, 'model_license_name', None),
            )
    total_weights_mb = sum(weight_estimates.values())

    # ── 2. Preprocessed frame/image storage on GPU ──────────────────────
    gpu_specs = _collect_gpu_preprocess_specs(ai_models)
    per_frame_mb = sum(_estimate_preprocess_tensor_mb(size, hp)
                       for size, hp in gpu_specs)
    queued_sets = max(max_pending_frames, 1)
    preprocess_storage_mb = per_frame_mb * queued_sets

    # ── 3. Available VRAM for batch activations ─────────────────────────
    available_mb = (total_vram_mb - total_weights_mb
                    - preprocess_storage_mb - overhead_mb)

    logger.info(
        f"VRAM budget: total={total_vram_mb:.0f}MB, "
        f"weights={total_weights_mb:.0f}MB ({len(ai_models)} models), "
        f"preprocess_queue={preprocess_storage_mb:.0f}MB "
        f"({queued_sets} sets × {per_frame_mb:.1f}MB/set, "
        f"{len(gpu_specs)} specs), "
        f"overhead={overhead_mb}MB, "
        f"available={available_mb:.0f}MB"
    )

    if available_mb <= 0:
        logger.warning("VRAM budget: no headroom for batching, forcing batch_size=1 "
                        "for all VRAM-scaled models")

    # ── 4. Compute per-model raw batch sizes ────────────────────────────
    # Only models with calibrated activation_per_item_mb are dynamically
    # scaled. batch_size_per_VRAM_GB is deprecated and ignored here.
    raw_batches = {}  # model id → raw batch size
    dynamic_models = []

    for model in ai_models:
        act_per_item = getattr(model, 'activation_per_item_mb', None)

        # Skip models without calibrated activation data (e.g. face models)
        if act_per_item is None or act_per_item <= 0:
            logger.info(
                f"VRAM budget: {model.model_file_name} → batch_size="
                f"{model.max_batch_size} (fixed from config, "
                f"weight={weight_estimates[id(model)]:.0f}MB)"
            )
            continue

        dynamic_models.append(model)

        if available_mb <= 0:
            raw_batches[id(model)] = 1
        else:
            batch_size = int(_UTILIZATION_FACTOR * available_mb / act_per_item)
            raw_batches[id(model)] = max(1, batch_size)

    if not dynamic_models:
        return

    # ── 5. Pipeline equalization ────────────────────────────────────────
    # All dynamic models share the same image flow through the pipeline.
    # The bottleneck model (highest activation per item, lowest raw batch)
    # determines the effective throughput. Setting all models to the same
    # batch size prevents fill_to_batch from padding with unnecessary zeros.
    pipeline_batch = min(raw_batches.values())
    pipeline_batch = _round_to_8(pipeline_batch)

    # Find which model is the bottleneck for logging
    bottleneck = min(dynamic_models, key=lambda m: raw_batches[id(m)])

    logger.info(
        f"VRAM budget: pipeline batch={pipeline_batch} "
        f"(bottleneck={bottleneck.model_file_name}, "
        f"raw={raw_batches[id(bottleneck)]}, rounded to nearest 8)"
    )

    # ── 6. Apply pipeline batch with per-model caps ─────────────────────
    for model in dynamic_models:
        batch_size = pipeline_batch
        raw = raw_batches[id(model)]
        act_per_item = getattr(model, 'activation_per_item_mb', None)

        # Use the immutable config cap stored at init time, not max_model_batch_size
        # which may have been overwritten by the legacy bspv path.
        max_cap = getattr(model, '_config_max_batch_cap', None)
        if max_cap is not None and max_cap > 0:
            batch_size = min(batch_size, max_cap)

        capped = f", capped by config={max_cap}" if max_cap is not None and max_cap > 0 and max_cap < pipeline_batch else ""
        logger.info(
            f"VRAM budget: {model.model_file_name} → batch_size={batch_size} "
            f"(raw={raw}, pipeline={pipeline_batch}{capped}, "
            f"act_per_item={act_per_item}MB, weight={weight_estimates[id(model)]:.0f}MB)"
        )

        model.max_model_batch_size = batch_size
        model.max_batch_size = batch_size
        model.max_queue_size = batch_size

