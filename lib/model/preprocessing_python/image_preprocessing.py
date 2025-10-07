import json
import logging
import os
import queue
import re
import threading
from contextlib import suppress
from typing import Optional

import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision
import numpy as np

decord.bridge.set_bridge('torch')

_DEFFCODE_LOG_SUPPRESSIONS = (
    "Manually discarding `frame_format`",
    "Manually discarding `-size/-s`",
    "Manually disabling `-framerate/-r`",
    "No usable pixel-format defined. Switching to default",
    "Pipeline terminated successfully.",
    "Running DeFFcode Version",
)

_DEFFCODE_LOG_FILTER = None
_DEFFCODE_DECODER = None
_LOGGER = logging.getLogger(__name__)

def _ensure_deffcode_log_filter() -> None:
    global _DEFFCODE_LOG_FILTER
    if _DEFFCODE_LOG_FILTER is not None:
        return

    class _DeffcodeNoiseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return not any(suppression in message for suppression in _DEFFCODE_LOG_SUPPRESSIONS)

    _DEFFCODE_LOG_FILTER = _DeffcodeNoiseFilter()
    for logger_name in ("FFdecoder", "Utilities", "Sourcer"):
        logging.getLogger(logger_name).addFilter(_DEFFCODE_LOG_FILTER)


def _get_deffcode_decoder():
    global _DEFFCODE_DECODER
    if _DEFFCODE_DECODER is None:
        #_ensure_deffcode_log_filter()
        from deffcode import FFdecoder as _ImportedDecoder  # type: ignore[import]

        _DEFFCODE_DECODER = _ImportedDecoder
    return _DEFFCODE_DECODER


_ensure_deffcode_log_filter()

def get_normalization_config(index, device):
    normalization_configs = [
        (torch.tensor([0.485, 0.456, 0.406], device=device), torch.tensor([0.229, 0.224, 0.225], device=device)),
        (torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device), torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)),
    ]
    return normalization_configs[index]

def custom_round(x, base=1):
    return base * round(x/base)


def _ensure_torch_tensor(frame, device: torch.device) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame
    elif hasattr(frame, "torch"):
        tensor = frame.torch()
    elif hasattr(frame, "to_torch"):
        tensor = frame.to_torch()
    else:
        array = np.asarray(frame)
        if not array.flags.writeable:
            array = np.copy(array)
        tensor = torch.from_numpy(array)

    if device is not None:
        tensor = tensor.to(device)
    return tensor

#TODO: SEE WHICH IS BETTER
def get_video_duration_torchvision(video_path):
    video = torchvision.io.VideoReader(video_path, "video")
    metadata = video.get_metadata()
    duration = metadata['video']['duration'][0]
    return duration

def get_video_duration_decord(video_path):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    num_frames = len(vr)
    frame_rate = vr.get_avg_fps()
    duration = num_frames / frame_rate
    return duration

def get_frame_transforms(use_half_precision, mean, std, vr_video=False, img_size=512):
    dtype = torch.float16 if use_half_precision else torch.float32
    transforms_list = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToDtype(dtype, scale=True),
    ]

    if mean is not None and std is not None:
        transforms_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transforms_list)

def vr_permute(frame):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio > 1.5:
        # 180 VR video, crop the right half of the image
        frame = frame[:, frame.shape[1]//2:]
    else:
        # 360 VR video, take the top half and the center 50% of the frame
        frame = frame[:frame.shape[0]//2, frame.shape[1]//4:3*frame.shape[1]//4]
    return frame

def preprocess_image(image_path, img_size=512, use_half_precision=True, device=None, norm_config=1):
    if device:
        device = torch.device(device)
    else:
        #Use CPU for Apple Silicon as well, because it cannot handle BICUBIC
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config, device)
    if (use_half_precision):
        imageTransforms = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToDtype(torch.float16, scale=True),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        imageTransforms = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
        ])
    return imageTransforms(read_image(image_path).to(device))
    

def _prepare_frame(frame, device, vr_video, frame_transforms):
    tensor = _ensure_torch_tensor(frame, device)
    if vr_video:
        tensor = vr_permute(tensor)
    tensor = tensor.permute(2, 0, 1)
    return frame_transforms(tensor)


#TODO: TRY OTHER PREPROCESSING METHODS AND TRY MAKING PREPROCESSING TRUE ASYNC
def preprocess_video(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
):
    if device:
        device = torch.device(device)
    else:
        #Use CPU for Apple Silicon as well, because it cannot handle BICUBIC
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)
    vr = None
    # Decode at source resolution and rely on PyTorch resizing so we mirror the training image pipeline
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps()) or 30.0
    frame_step = 1
    if frame_interval and frame_interval > 0:
        frame_step = max(1, round(fps * frame_interval))

    processed = 0
    for i in range(0, len(vr), frame_step):
        frame_tensor = _prepare_frame(vr[i], device, vr_video, frame_transforms)
        if use_timestamps:
            if frame_interval and frame_interval > 0:
                frame_index = processed * frame_interval
            else:
                frame_index = i / fps if fps else float(processed)
        else:
            frame_index = i
        yield (frame_index, frame_tensor)
        processed += 1
    del vr

def _parse_fps_value(value, default: float = 30) -> float:
    if value is None:
        return default

    try:
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if numerator is not None and denominator is not None:
            denominator = denominator or 1
            return float(numerator) / float(denominator)

        if isinstance(value, (list, tuple)):
            if not value:
                return default
            if len(value) == 2 and value[1]:
                return float(value[0]) / float(value[1])
            return _parse_fps_value(value[0], default)

        if isinstance(value, dict):
            num = value.get("numerator")
            den = value.get("denominator") or 1
            if num is not None:
                return float(num) / float(den)

        return float(value)
    except Exception:
        return default


_DEFFCODE_HW_DECODER_MAP = {
    "h264": "h264_cuvid",
    "avc": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "mpeg2": "mpeg2_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
}

_UNSUPPORTED_NVDEC_PIXEL_FORMATS = {
    "yuv444p",
    "yuv444p10le",
    "yuv444p12le",
    "yuvj444p",
}


def _guess_deffcode_hw_decoder(codec_name: Optional[str]) -> Optional[str]:
    if not codec_name:
        return None
    lowered = codec_name.lower()
    sanitized = lowered.replace("_", "").replace("-", "")
    if sanitized.startswith("h264") or sanitized.startswith("avc"):
        return _DEFFCODE_HW_DECODER_MAP["h264"]
    if sanitized.startswith("hevc") or sanitized.startswith("h265"):
        return _DEFFCODE_HW_DECODER_MAP["hevc"]
    for key, value in _DEFFCODE_HW_DECODER_MAP.items():
        if sanitized.startswith(key):
            return value
    return _DEFFCODE_HW_DECODER_MAP.get(lowered)

def preprocess_video_deffcode(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
):

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)

    decoder_kwargs = {}
    if not vr_video:
        decoder_kwargs["-custom_resolution"] = "null"

    decoder_cls = _get_deffcode_decoder()
    decoder = decoder_cls(video_path, frame_format="rgb24", **decoder_kwargs).formulate()

    try:
        metadata = json.loads(decoder.metadata)
    except Exception:
        metadata = {}

    source_fps = _parse_fps_value(metadata.get("source_video_framerate")) if metadata else None
    output_fps = _parse_fps_value(metadata.get("output_framerate")) if metadata else None
    effective_fps = source_fps or output_fps or 30.0

    frame_step_frames = 1
    if frame_interval and frame_interval > 0:
        if source_fps:
            frame_step_frames = max(1, round(source_fps * frame_interval))
        elif output_fps:
            frame_step_frames = max(1, round(output_fps * frame_interval))
        else:
            frame_step_frames = max(1, round(1.0 / frame_interval))

    processed = 0

    try:
        for index, frame in enumerate(decoder.generateFrame()):
            if frame is None:
                continue

            if frame_interval and frame_interval > 0 and (index % frame_step_frames) != 0:
                continue

            tensor = _prepare_frame(frame, device, vr_video, frame_transforms)

            if use_timestamps:
                if frame_interval and frame_interval > 0:
                    output_index = processed * frame_interval
                else:
                    output_index = index / (effective_fps or 1.0)
            else:
                output_index = index

            yield (output_index, tensor)
            processed += 1
    finally:
        terminate = getattr(decoder, "terminate", None)
        if callable(terminate):
            terminate()


def preprocess_video_deffcode_gpu(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
):

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        raise RuntimeError("CUDA device unavailable for DeFFcode GPU preprocessing")

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)

    # Probe metadata to determine the appropriate NVDEC decoder
    decoder_cls = _get_deffcode_decoder()
    probe_decoder = decoder_cls(video_path, frame_format="null").formulate()
    try:
        metadata = json.loads(probe_decoder.metadata)
    finally:
        terminate = getattr(probe_decoder, "terminate", None)
        if callable(terminate):
            terminate()

    source_pixfmt = (metadata.get("source_video_pixfmt") or "").lower()
    source_fps = _parse_fps_value(metadata.get("source_video_framerate")) if metadata else None
    if source_pixfmt in _UNSUPPORTED_NVDEC_PIXEL_FORMATS:
        raise RuntimeError(
            f"NVDEC does not support pixel format '{source_pixfmt}'. Use the CPU DeFFcode backend or re-encode the source."
        )

    env_hw_decoder = os.environ.get("DEFFCODE_HW_DECODER")
    hw_decoder = env_hw_decoder or _guess_deffcode_hw_decoder(metadata.get("source_video_decoder"))
    if not hw_decoder:
        raise RuntimeError(
            "Unable to determine an appropriate NVDEC hardware decoder for this video stream"
        )

    device_index = None
    if device.index is not None:
        device_index = device.index
    elif torch.cuda.device_count() > 0:
        device_index = torch.cuda.current_device()

    ffprefixes = ["-vsync", "0", "-hwaccel", "cuda"]
    if device_index is not None:
        ffprefixes.extend(["-hwaccel_device", str(device_index)])

    frame_step_frames = 1
    if frame_interval and frame_interval > 0:
        if source_fps:
            frame_step_frames = max(1, round(source_fps * frame_interval))
        else:
            frame_step_frames = max(1, round(1.0 / frame_interval))

    vf_filters = []
    vf_filters.append(f"select=not(mod(n\\,{frame_step_frames}))")
    vf_filters.extend(
        [
            "format=nv12",
            "scale=in_range=tv:out_range=pc:flags=bicubic",
            "format=rgb24",
        ]
    )
    decoder_kwargs = {
        "-vcodec": hw_decoder,
        "-enforce_cv_patch": True,
        "-ffprefixes": ffprefixes,
        "-custom_resolution": "null",
        "-framerate": "null",
    }

    if vf_filters:
        decoder_kwargs["-vf"] = ",".join(vf_filters)
    decoder_cls = _get_deffcode_decoder()
    decoder = decoder_cls(
        video_path,
        frame_format="rgb24",
        **decoder_kwargs,
    ).formulate()

    try:
        try:
            runtime_meta = json.loads(decoder.metadata)
        except Exception:
            runtime_meta = metadata

        effective_fps = None
        if source_fps and frame_step_frames:
            effective_fps = source_fps / frame_step_frames

        if runtime_meta and not effective_fps:
            effective_fps = _parse_fps_value(runtime_meta.get("output_framerate"))
            if not effective_fps:
                effective_fps = _parse_fps_value(runtime_meta.get("source_video_framerate"))
        if not effective_fps:
            effective_fps = _parse_fps_value(metadata.get("source_video_framerate")) if metadata else 30.0

        processed = 0
        for index, frame in enumerate(decoder.generateFrame()):
            if frame is None:
                continue

            tensor = _prepare_frame(frame, device, vr_video, frame_transforms)

            if use_timestamps:
                if frame_interval and frame_interval > 0:
                    output_index = processed * frame_interval
                else:
                    output_index = index / (effective_fps or 1.0)
            else:
                if frame_interval and frame_interval > 0:
                    output_index = processed * frame_step_frames
                else:
                    output_index = index

            yield (output_index, tensor)
            processed += 1

        if processed == 0:
            _LOGGER.warning(
                "DeFFcode CUDA pipeline produced zero frames for '%s'; falling back to CPU backend.",
                video_path,
            )
    finally:
        terminate = getattr(decoder, "terminate", None)
        if callable(terminate):
            terminate()

    if processed == 0:
        cpu_generator = preprocess_video_deffcode(
            video_path,
            frame_interval=frame_interval,
            img_size=img_size,
            use_half_precision=use_half_precision,
            device=device,
            use_timestamps=use_timestamps,
            vr_video=vr_video,
            norm_config=norm_config,
        )
        for payload in cpu_generator:
            yield payload