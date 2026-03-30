
import time

import numpy as np
import torch
import torch.nn.functional as F

from lib.model.ai_model import AIModel


ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class AIFaceEmbeddingModel(AIModel):
    """AI model subclass for face embedding (ArcFace via torch.export)."""

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)
        self.fill_to_batch = False
        self.model_return_confidence = configValues.get("model_return_confidence", True)

    async def worker_function(self, data):
        batch_started_at = time.time()
        for item in data:
            item_future = item.item_future
            try:
                embedding = self._run_embedding_item(item)
                await item_future.set_data(item.output_names[0], embedding)
            except Exception as e:
                self.logger.error(f"Error in AIFaceEmbeddingModel: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                item_future.set_exception(e)
        self.logger.debug(
            f"Processed {len(data)} images in {time.time() - batch_started_at} in {self.model_file_name} ({self.model_category})"
        )

    async def load(self):
        await super().load()

    def _run_embedding_item(self, item):
        item_future = item.item_future
        tensor = item_future[item.input_names[0]]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Embedding face model expects tensor input")

        aligned_image = self._try_aligned_face_image(item_future)
        # Eagerly release the full-resolution source tensor from this
        # child future now that alignment has been resolved.  This allows
        # GC to free the large tensor as soon as all sibling regions for
        # the same frame have finished.
        _clear_region_source(item_future)
        if aligned_image is not None:
            image = aligned_image
        else:
            image = _ensure_model_rgb_gpu(tensor)

        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False)
        input_tensor = (image - 127.5) / 127.5
        input_tensor = input_tensor.to(self.device)

        # Run through secure ModelRunner — model never exposed
        outputs = self.model.run_raw_multi_output(input_tensor)
        feat_np = outputs[0].reshape(-1)
        norm = float(np.linalg.norm(feat_np))
        return [{"vector": feat_np.tolist(), "norm": norm, "embedder": self.model_file_name}]

    def _try_aligned_face_image(self, item_future):
        try:
            region_target = _resolve_future_key(item_future, "dynamic_region_target")
            source_tensor = _resolve_future_key(item_future, "dynamic_region_source")
            if not isinstance(region_target, dict) or not isinstance(source_tensor, torch.Tensor):
                return None

            metadata = region_target.get("metadata") or {}
            kps = metadata.get("kps")
            kps = _normalize_kps(kps)
            if kps is None:
                return None

            return _norm_crop_gpu(source_tensor, kps, image_size=112)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Embedding / alignment helpers (moved from face_torch_export_model.py)
# ---------------------------------------------------------------------------

def _clear_region_source(item_future):
    """Eagerly release the full-resolution source tensor from a child
    ItemFuture so the large tensor can be garbage-collected as soon as
    all sibling regions for the same frame have finished."""
    data = getattr(item_future, 'data', None)
    if data is None:
        return
    for key in list(data.keys()):
        if isinstance(key, str) and key.startswith("dynamic_region_source"):
            data[key] = None


def _ensure_model_rgb_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [0, 255] float RGB CHW, staying on the same device."""
    t = tensor.detach()
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        raise ValueError("Expected CHW tensor")
    t = t.float()
    tmin = float(t.min())
    tmax = float(t.max())
    if tmin >= 0.0 and tmax > 1.1:
        return t.clamp(0.0, 255.0)
    if tmin >= -1.1 and tmax <= 1.1:
        return ((t + 1.0) * 127.5).clamp(0.0, 255.0)
    if tmin >= -5.0 and tmax <= 5.0:
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=t.device, dtype=torch.float32).view(3, 1, 1)
        return ((t * std + mean) * 255.0).clamp(0.0, 255.0)
    if tmin >= 0.0 and tmax <= 1.1:
        return (t * 255.0).clamp(0.0, 255.0)
    return t.clamp(0.0, 255.0)


def _resolve_future_key(item_future, prefix):
    """Look up *prefix* in an ``ItemFuture``, falling back to any key that
    starts with *prefix* (e.g. ``dynamic_region_target__<alias>`` when the
    exact key ``dynamic_region_target`` is absent)."""
    value = item_future[prefix]
    if value is not None:
        return value
    data = getattr(item_future, "data", None)
    if data is None:
        return None
    for key in data:
        if key.startswith(prefix):
            return data[key]
    return None


def _normalize_kps(kps):
    if kps is None:
        return None
    arr = np.asarray(kps, dtype=np.float32)
    if arr.shape != (5, 2):
        return None
    return arr


def _estimate_arcface_affine(landmark: np.ndarray, image_size: int = 112):
    if landmark.shape != (5, 2):
        raise ValueError("Expected landmark shape (5, 2)")

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = ARCFACE_DST.copy() * ratio
    dst[:, 0] += diff_x

    src = landmark.astype(np.float32)
    dst = dst.astype(np.float32)

    a_rows = []
    b_vals = []
    for (x, y), (u, v) in zip(src, dst):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0])
        b_vals.append(u)
        b_vals.append(v)

    A = np.asarray(a_rows, dtype=np.float32)
    b = np.asarray(b_vals, dtype=np.float32)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.asarray(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
        ],
        dtype=np.float32,
    )
    return M


def _norm_crop_gpu(source_tensor: torch.Tensor, landmark: np.ndarray, image_size: int = 112):
    """GPU-based ArcFace alignment using grid_sample (no CPU roundtrip)."""
    M = _estimate_arcface_affine(landmark, image_size=image_size)

    t = source_tensor.detach()
    if t.dim() == 4:
        t = t[0]
    t = t.float()
    _, iH, iW = t.shape

    # Invert affine: maps output pixel coords -> input pixel coords
    M3 = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    M_inv = np.linalg.inv(M3)[:2].astype(np.float64)

    # Convert pixel-coord affine to normalised [-1, 1] coords for grid_sample
    oW = oH = float(image_size)
    theta = np.zeros((2, 3), dtype=np.float64)
    theta[0, 0] = M_inv[0, 0] * (oW - 1) / (iW - 1)
    theta[0, 1] = M_inv[0, 1] * (oH - 1) / (iW - 1)
    theta[0, 2] = (M_inv[0, 0] * (oW - 1) + M_inv[0, 1] * (oH - 1) + 2 * M_inv[0, 2]) / (iW - 1) - 1
    theta[1, 0] = M_inv[1, 0] * (oW - 1) / (iH - 1)
    theta[1, 1] = M_inv[1, 1] * (oH - 1) / (iH - 1)
    theta[1, 2] = (M_inv[1, 0] * (oW - 1) + M_inv[1, 1] * (oH - 1) + 2 * M_inv[1, 2]) / (iH - 1) - 1

    theta_t = torch.tensor(theta, dtype=torch.float32, device=t.device).unsqueeze(0)
    grid = F.affine_grid(theta_t, [1, 3, image_size, image_size], align_corners=True)
    aligned = F.grid_sample(t.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return aligned.squeeze(0)
