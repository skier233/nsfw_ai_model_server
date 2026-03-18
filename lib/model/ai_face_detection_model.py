
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image

from lib.model.ai_model import AIModel


class AIFaceDetectionModel(AIModel):
    """AI model subclass for face detection (SCRFD via torch.export)."""

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)
        self.fill_to_batch = False
        self.model_threshold = configValues.get("model_threshold", 0.5)
        self.det_size = tuple(configValues.get("det_size", [640, 640]))
        self.det_nms_thresh = float(configValues.get("det_nms_thresh", 0.4))

    async def worker_function(self, data):
        batch_started_at = time.time()
        for item in data:
            item_future = item.item_future
            try:
                detections = self._run_detection_item(item)
                await item_future.set_data(item.output_names[0], detections)
            except Exception as e:
                self.logger.error(f"Error in AIFaceDetectionModel: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                item_future.set_exception(e)
        self.logger.debug(
            f"Processed {len(data)} images in {time.time() - batch_started_at} in {self.model_file_name} ({self.model_category})"
        )

    async def load(self):
        await super().load()

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


# ---------------------------------------------------------------------------
# Detection helpers (moved from face_torch_export_model.py)
# ---------------------------------------------------------------------------

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


def _run_module(model_runner, input_tensor):
    """Run model via the secure ModelRunner/PythonModel interface.
    Returns list of numpy arrays (one per output head)."""
    return model_runner.run_raw_multi_output(input_tensor)


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
