"""
DEPRECATED: This module has been replaced by:
  - lib.model.ai_face_detection_model.AIFaceDetectionModel
  - lib.model.ai_face_embedding_model.AIFaceEmbeddingModel

Kept for reference only. No longer imported by model_manager.
"""

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image

from lib.model.ai_model import AIModel
from lib.utils.torch_device_selector import get_device_string


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


class FaceTorchExportModel(AIModel):
    def __init__(self, configValues):
        self.logger = logging.getLogger("logger")
        self.max_queue_size = configValues.get("max_queue_size", None)
        self.max_batch_size = configValues.get("max_batch_size", 1)
        self.instance_count = configValues.get("instance_count", 1)
        self.max_batch_waits = configValues.get("max_batch_waits", -1)
        timeout_ms = configValues.get("batch_collect_timeout_ms", 10)
        if timeout_ms is None:
            timeout_ms = 0
        self.batch_collect_timeout = max(timeout_ms, 0) / 1000.0

        self.model_file_name = configValues.get("model_file_name", None)
        self.model_threshold = configValues.get("model_threshold", 0.5)
        self.model_return_confidence = configValues.get("model_return_confidence", True)
        self.model_category = configValues.get("model_category", ["face"])
        self.model_type = configValues.get("model_type", "FaceTorchExport")
        self.model_version = configValues.get("model_version", 1.0)
        self.model_identifier = configValues.get("model_identifier", None)
        self.model_image_size = configValues.get("model_image_size", 640)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.model_capabilities = configValues.get("model_capabilities", ["detection"])
        self.supported_target_scopes = configValues.get("supported_target_scopes", ["asset", "frame", "region"])

        if isinstance(self.model_category, str):
            self.model_category = [self.model_category]
        if isinstance(self.model_capabilities, str):
            self.model_capabilities = [self.model_capabilities]
        if isinstance(self.supported_target_scopes, str):
            self.supported_target_scopes = [self.supported_target_scopes]

        if self.model_file_name is None:
            raise ValueError("model_file_name is required for face_torch_export models")

        self.face_model_role = str(configValues.get("face_model_role", "detection")).lower()
        self.det_size = tuple(configValues.get("det_size", [640, 640]))
        self.det_nms_thresh = float(configValues.get("det_nms_thresh", 0.4))
        self.device = configValues.get("device", None)
        if self.device is None:
            self.device = get_device_string()

        self.model = None

    async def worker_function(self, data):
        batch_started_at = time.time()
        for item in data:
            item_future = item.item_future
            try:
                if self.face_model_role == "detection":
                    detections = self._run_detection_item(item)
                    await item_future.set_data(item.output_names[0], detections)
                elif self.face_model_role == "embedding":
                    embedding = self._run_embedding_item(item)
                    await item_future.set_data(item.output_names[0], embedding)
                else:
                    raise ValueError(f"Unsupported face_model_role '{self.face_model_role}'")
            except Exception as e:
                self.logger.error(f"Error in FaceTorchExportModel ({self.face_model_role}): {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                item_future.set_exception(e)
        self.logger.debug(
            f"Processed {len(data)} images in {time.time() - batch_started_at} in {self.model_file_name} ({self.model_category})"
        )

    async def load(self):
        if self.model is not None:
            return
        model_path = Path("./models") / f"{self.model_file_name}.pt2"
        if not model_path.exists():
            raise FileNotFoundError(f"Face torch.export model not found: {model_path}")
        exported_program = torch.export.load(str(model_path))
        self.model = exported_program.module().to(self.device)
        self.logger.info(
            f"Loaded face torch.export model: {model_path} (role={self.face_model_role}, device={self.device})"
        )

    def _run_detection_item(self, item):
        item_future = item.item_future
        threshold = item_future[item.input_names[1]]
        if threshold is None:
            threshold = self.model_threshold

        image = self._resolve_image_for_detection(item)
        det, kpss = run_detection(
            self.model,
            image,
            det_size=self.det_size,
            det_thresh=float(threshold),
            nms_thresh=self.det_nms_thresh,
            device=self.device,
        )

        output = []
        for index, det_row in enumerate(det):
            bbox = [float(det_row[0]), float(det_row[1]), float(det_row[2]), float(det_row[3])]
            score = float(det_row[4])
            entry = {"bbox": bbox, "score": score, "detector": self.model_file_name}
            if kpss is not None and index < len(kpss):
                entry["kps"] = [[float(x), float(y)] for x, y in kpss[index]]
            output.append(entry)
        return output

    def _run_embedding_item(self, item):
        item_future = item.item_future
        tensor = item_future[item.input_names[0]]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Embedding face model expects tensor input")

        aligned_image = self._try_aligned_face_image(item_future)
        if aligned_image is not None:
            image = aligned_image
        else:
            image = _ensure_model_rgb_gpu(tensor)

        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False)
        input_tensor = (image - 127.5) / 127.5
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            feat = self.model(input_tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        feat_np = feat.detach().cpu().numpy().reshape(-1)
        norm = float(np.linalg.norm(feat_np))
        return [{"vector": feat_np.tolist(), "norm": norm, "embedder": self.model_file_name}]

    def _try_aligned_face_image(self, item_future):
        try:
            region_target = _resolve_future_key(item_future, "dynamic_region_target")
            source_tensor = _resolve_future_key(item_future, "dynamic_region_source")
            if not isinstance(region_target, dict) or not isinstance(source_tensor, torch.Tensor):
                return None

            metadata = region_target.get("metadata") or {}
            detector_payload = metadata.get("detector_payload") or {}
            kps = detector_payload.get("kps")
            kps = _normalize_kps(kps)
            if kps is None:
                return None

            return _norm_crop_gpu(source_tensor, kps, image_size=112)
        except Exception:
            return None

    def _resolve_image_for_detection(self, item):
        item_future = item.item_future
        source = item_future[item.input_names[0]]
        if isinstance(source, torch.Tensor):
            return source

        image_path = item_future["image_path"]
        if image_path is not None:
            image = read_image(str(image_path))
            return image
        raise ValueError("Detection model could not resolve source image")


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


def tensor_to_model_rgb(tensor: torch.Tensor):
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        raise ValueError("Expected CHW tensor")

    arr = t.float().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    min_val = float(arr.min())
    max_val = float(arr.max())

    if min_val >= -1.1 and max_val <= 1.1:
        arr = (arr + 1.0) * 127.5
    elif min_val >= -5.0 and max_val <= 5.0:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr * std + mean) * 255.0
    elif min_val >= 0.0 and max_val <= 1.1:
        arr = arr * 255.0

    arr = np.clip(arr, 0.0, 255.0)
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()


def _resolve_future_key(item_future, prefix):
    """Look up *prefix* in an ``ItemFuture``, falling back to any key that
    starts with *prefix* (e.g. ``dynamic_region_target__<alias>`` when the
    exact key ``dynamic_region_target`` is absent).  This lets models work
    with both aliased (v3 dynamic pipelines) and un-aliased (standalone
    region pipelines) key names."""
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

    # Invert affine: maps output pixel coords → input pixel coords
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


def _norm_crop_tensor(source_tensor: torch.Tensor, landmark: np.ndarray, image_size: int = 112):
    """Legacy CPU-based alignment (kept for fallback compatibility)."""
    from PIL import Image

    M = _estimate_arcface_affine(landmark, image_size=image_size)
    M3 = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    inv = np.linalg.inv(M3)
    coeffs = inv[:2, :].reshape(-1).tolist()

    rgb = tensor_to_model_rgb(source_tensor)
    chw = rgb.detach().cpu().numpy().astype(np.float32)
    hwc = np.transpose(chw, (1, 2, 0))
    hwc_u8 = np.clip(hwc, 0.0, 255.0).astype(np.uint8)

    pil = Image.fromarray(hwc_u8, mode="RGB")
    aligned = pil.transform((image_size, image_size), Image.AFFINE, data=coeffs, resample=Image.BILINEAR)
    aligned_arr = np.asarray(aligned, dtype=np.float32)
    aligned_chw = np.transpose(aligned_arr, (2, 0, 1))
    return torch.from_numpy(aligned_chw)


def _scrfd_meta(outputs_len):
    if outputs_len == 6:
        return 3, [8, 16, 32], 2, False
    if outputs_len == 9:
        return 3, [8, 16, 32], 2, True
    if outputs_len == 10:
        return 5, [8, 16, 32, 64, 128], 1, False
    if outputs_len == 15:
        return 5, [8, 16, 32, 64, 128], 1, True
    raise ValueError(f"Unexpected SCRFD output count: {outputs_len}")


def _run_module(module, input_tensor):
    with torch.no_grad():
        outputs = module(input_tensor)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    elif not isinstance(outputs, (tuple, list)):
        outputs = tuple(outputs)
    return [o.detach().cpu().numpy() for o in outputs]


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, thresh):
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = [dets[:, i] for i in range(5)]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def run_detection(det_model, img, det_size=(640, 640), det_thresh=0.5, nms_thresh=0.4, device="cpu"):
    # Detect whether input is already in raw [0-255] RGB range (GPU-resident).
    # If so, skip the expensive CPU-based tensor_to_model_rgb conversion.
    img_t = img.detach() if isinstance(img, torch.Tensor) else img
    if isinstance(img_t, torch.Tensor) and img_t.dim() >= 3:
        tmin = float(img_t.min())
        tmax = float(img_t.max())
        is_raw = tmin >= 0.0 and tmax > 1.1 and tmax <= 256.0
    else:
        is_raw = False

    if is_raw:
        rgb_tensor = img_t.float()
        if rgb_tensor.dim() == 4:
            rgb_tensor = rgb_tensor[0]
    else:
        rgb_tensor = tensor_to_model_rgb(img)

    im_ratio = float(rgb_tensor.shape[1]) / rgb_tensor.shape[2]
    model_ratio = float(det_size[1]) / det_size[0]
    if im_ratio > model_ratio:
        new_height = det_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = det_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / rgb_tensor.shape[1]

    resized = F.interpolate(
        rgb_tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    canvas = torch.zeros((3, det_size[1], det_size[0]), dtype=torch.float32, device=rgb_tensor.device)
    canvas[:, :new_height, :new_width] = resized
    input_tensor = ((canvas - 127.5) / 128.0).unsqueeze(0).to(device)
    outputs = _run_module(det_model, input_tensor)

    fmc, feat_strides, num_anchors, use_kps = _scrfd_meta(len(outputs))
    batched = outputs[0].ndim == 3
    input_height, input_width = int(input_tensor.shape[2]), int(input_tensor.shape[3])

    scores_list, bboxes_list, kpss_list = [], [], []
    for idx, stride in enumerate(feat_strides):
        if batched:
            scores = outputs[idx][0]
            bbox_preds = outputs[idx + fmc][0] * stride
            if use_kps:
                kps_preds = outputs[idx + fmc * 2][0] * stride
        else:
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride
            if use_kps:
                kps_preds = outputs[idx + fmc * 2] * stride

        height = input_height // stride
        width = input_width // stride
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))

        pos_inds = np.where(scores >= det_thresh)[0]
        if pos_inds.size == 0:
            continue
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

    if not scores_list:
        return np.zeros((0, 5), dtype=np.float32), None

    scores = np.vstack(scores_list)
    bboxes = np.vstack(bboxes_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    order = pre_det[:, 4].argsort()[::-1]
    pre_det = pre_det[order, :]
    keep = nms(pre_det, nms_thresh)
    det = pre_det[keep, :]

    if use_kps:
        kpss = np.vstack(kpss_list) / det_scale
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]
    else:
        kpss = None

    return det, kpss
