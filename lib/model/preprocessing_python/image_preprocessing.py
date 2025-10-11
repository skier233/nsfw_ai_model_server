import json
import logging
import os
import queue
import re
import threading
from contextlib import suppress
from typing import Optional

import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision
import numpy as np
from deffcode import Sourcer

try:
    import decord
    decord.bridge.set_bridge('torch')
    _HAS_DECORD = True
except Exception:  # pragma: no cover - optional dependency
    decord = None
    _HAS_DECORD = False

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


def _ensure_torch_tensor(
    frame,
    device: torch.device,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame
    elif hasattr(frame, "torch"):
        tensor = frame.torch()
    elif hasattr(frame, "to_torch"):
        tensor = frame.to_torch()
    else:
        array = np.asarray(frame)
        copy = np.copy(array)
        tensor = torch.from_numpy(copy)

    if device is not None:
        if pin_memory and tensor.device.type == "cpu" and torch.cuda.is_available():
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=non_blocking)
    return tensor

#TODO: SEE WHICH IS BETTER
def get_video_duration_torchvision(video_path):
    video = torchvision.io.VideoReader(video_path, "video")
    metadata = video.get_metadata()
    duration = metadata['video']['duration'][0]
    return duration

def get_video_duration_deffcode(video_path):

    sourcer = Sourcer(video_path).probe_stream()
    try:
        metadata = sourcer.retrieve_metadata()
        # Prefer an explicit duration field if available
        duration = metadata.get("source_duration_sec")
        if duration is not None:
            try:
                return float(duration)
            except Exception:
                pass

        # Fall back to frame-count / frame-rate
        num_frames = metadata.get("approx_video_nframes") or metadata.get("source_video_num_frames") or 0
        try:
            num_frames = float(num_frames)
        except Exception:
            num_frames = 0.0

        frame_rate = _parse_fps_value(metadata.get("source_video_framerate"), default=30.0) or 30.0
        duration = (num_frames / frame_rate) if frame_rate and num_frames else 0.0
        return duration
    finally:
        terminate = getattr(sourcer, "terminate", None)
        if callable(terminate):
            terminate()

def get_video_duration_decord(video_path):
    if not _HAS_DECORD:
        raise RuntimeError("decord is not installed; install decord to use this function")
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    num_frames = len(vr)
    frame_rate = vr.get_avg_fps()
    duration = num_frames / frame_rate
    return duration

def get_frame_transforms(
    use_half_precision,
    mean,
    std,
    vr_video: bool = False,
    img_size: int = 512,
    apply_resize: bool = True,
):
    dtype = torch.float16 if use_half_precision else torch.float32
    transforms_list = [
        transforms.ToDtype(dtype, scale=True),
    ]
    if apply_resize:
        transforms_list.insert(0, transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC))

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

class DimensionError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return f"DimensionError({super().__str__()})"

def preprocess_image(image_path, img_size=512, use_half_precision=True, device=None, norm_config=1):
    # Resolve device
    if device:
        device = torch.device(device)
    else:
        # Use CPU for Apple Silicon as well, because it cannot handle BICUBIC
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std = get_normalization_config(norm_config, device)

    # Build the canonical transforms (resize -> dtype -> normalize)
    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=False,
        img_size=img_size,
        apply_resize=True,
    )

    # Read image using torchvision (returns a tensor). We explicitly validate
    # dimensionality to catch animated GIFs, grayscale, or unexpected channel counts
    img = read_image(image_path)
    # Move to target device and run transforms
    img = img.to(device)
    out = frame_transforms(img)

    # Validate final tensor shape: (3, img_size, img_size)
    if out.ndim != 3 or out.shape[0] != 3:
        raise DimensionError(
            f"Invalid Image: Image has invalid shape {tuple(out.shape)} for '{image_path}'; "
            f"expected (3, {img_size}, {img_size})."
        )

    return out
    

def _prepare_frame(frame, device, vr_video, frame_transforms):
    tensor = _ensure_torch_tensor(frame, device, pin_memory=True, non_blocking=True)
    
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
    """
    Preprocess video using decord (CPU-based).
    Falls back to deffcode if decord is not available.
    """
    if not _HAS_DECORD:
        _LOGGER.warning("decord not available, falling back to deffcode CPU preprocessing")
        yield from preprocess_video_deffcode(
            video_path=video_path,
            frame_interval=frame_interval,
            img_size=img_size,
            use_half_precision=use_half_precision,
            device=device,
            use_timestamps=use_timestamps,
            vr_video=vr_video,
            norm_config=norm_config,
        )
        return

    if device:
        device = torch.device(device)
    else:
        #Use CPU for Apple Silicon as well, because it cannot handle BICUBIC
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config, device)

    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=vr_video,  # Only apply resize for VR videos; decord handles it for non-VR
    )
    vr = None
    if vr_video:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    else:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), width=img_size, height=img_size)
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


def _target_fps(frame_interval: float) -> Optional[float]:
    if frame_interval is None or frame_interval <= 0:
        return None
    return max(1.0 / frame_interval, 0.001)


def _format_ffmpeg_float(value: float) -> str:
    text = f"{value:.12f}"
    text = text.rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text


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
    sanitized = re.sub(r"[\s\(\)\[\]{}]", "", lowered)
    sanitized = sanitized.replace("_", "").replace("-", "")
    if sanitized.startswith("h264") or sanitized.startswith("avc"):
        return _DEFFCODE_HW_DECODER_MAP["h264"]
    if sanitized.startswith("hevc") or sanitized.startswith("h265"):
        return _DEFFCODE_HW_DECODER_MAP["hevc"]
    for key, value in _DEFFCODE_HW_DECODER_MAP.items():
        if sanitized.startswith(key):
            return value
    return _DEFFCODE_HW_DECODER_MAP.get(lowered)

def preprocess_video_deffcode_auto(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
):
    """
    Automatically select GPU or CPU preprocessing based on video resolution and GPU availability.
    
    Strategy:
    - For high-resolution videos (4K+): Use GPU preprocessing to accelerate the bottleneck
    - For lower-resolution videos: Use CPU preprocessing to avoid GPU contention with AI models
    - Falls back gracefully if GPU preprocessing fails
    
    This approach maximizes overall throughput by using GPU resources where they provide 
    the most benefit while avoiding GPU contention when AI inference is the bottleneck.
    """
    if device:
        target_device = torch.device(device)
    else:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_gpu_preprocessing = False
    
    # Determine if GPU preprocessing should be used based on resolution
    if target_device.type == 'cuda':
        try:
            # Probe video metadata to check resolution using the higher-level Sourcer API
            sourcer = Sourcer(video_path).probe_stream()
            try:
                metadata = sourcer.retrieve_metadata()

                width = metadata.get("source_video_resolution", [0, 0])[0]
                height = metadata.get("source_video_resolution", [0, 0])[1]

                # Use GPU preprocessing for 4K+ videos (with small margin for non-standard resolutions)
                # 4K is typically 3840x2160, so we check for >= 3600 width or >= 1900 height
                if width >= 3600 or height >= 1900:
                    use_gpu_preprocessing = True
                    _LOGGER.debug(
                        "Video resolution %dx%d qualifies for GPU preprocessing",
                        width,
                        height,
                    )
                else:
                    _LOGGER.debug(
                        "Video resolution %dx%d will use CPU preprocessing to avoid GPU contention",
                        width,
                        height,
                    )
            finally:
                terminate = getattr(sourcer, "terminate", None)
                if callable(terminate):
                    terminate()
        except Exception as exc:
            _LOGGER.warning(
                "Failed to probe video resolution for '%s': %s. Defaulting to CPU preprocessing.",
                video_path,
                exc,
            )
            use_gpu_preprocessing = False

    # Try GPU preprocessing if determined appropriate
    if use_gpu_preprocessing:
        try:
            yield from preprocess_video_deffcode_gpu(
                video_path=video_path,
                frame_interval=frame_interval,
                img_size=img_size,
                use_half_precision=use_half_precision,
                device=target_device,
                use_timestamps=use_timestamps,
                vr_video=vr_video,
                norm_config=norm_config,
            )
            return
        except Exception as exc:
            _LOGGER.warning(
                "GPU preprocessing failed for '%s': %s. Falling back to CPU.",
                video_path,
                exc,
            )

    # Use CPU preprocessing (either by choice or as fallback)
    yield from preprocess_video_deffcode(
        video_path=video_path,
        frame_interval=frame_interval,
        img_size=img_size,
        use_half_precision=use_half_precision,
        device=target_device,
        use_timestamps=use_timestamps,
        vr_video=vr_video,
        norm_config=norm_config,
    )

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
    """
    Preprocess video using CPU-based DeFFcode with FFmpeg filtering.
    Uses FFmpeg select filter for efficient frame skipping.
    """
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std = get_normalization_config(norm_config, device)

    target_height: Optional[int]
    target_width: Optional[int]
    if isinstance(img_size, (tuple, list)) and len(img_size) >= 2:
        target_height = int(img_size[0])
        target_width = int(img_size[1])
    elif isinstance(img_size, (int, float)):
        size_int = int(img_size)
        target_height = size_int
        target_width = size_int
    else:
        target_height = None
        target_width = None

    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=True,  # Always use PyTorch resize in CPU mode
    )

    # Build FFmpeg filter chain
    decoder_kwargs = {
        "-ffprefixes": ["-vsync", "0"],  # Disable vsync for accurate frame selection
    }
    if not vr_video:
        decoder_kwargs["-custom_resolution"] = "null"

    # Probe metadata first to get FPS for frame-based selection
    decoder_cls = _get_deffcode_decoder()
    probe_decoder = decoder_cls(video_path, frame_format="null").formulate()
    try:
        metadata = json.loads(probe_decoder.metadata)
    except Exception:
        metadata = {}
    finally:
        terminate = getattr(probe_decoder, "terminate", None)
        if callable(terminate):
            terminate()

    source_fps = _parse_fps_value(metadata.get("source_video_framerate")) if metadata else None
    effective_fps = source_fps or 30.0

    # Add frame-based select filter to skip frames at decode time
    frame_step_frames = 1
    if frame_interval and frame_interval > 0 and source_fps:
        frame_step_frames = max(1, round(source_fps * frame_interval))
    
    if frame_step_frames > 1:
        vf_filter = f"select='not(mod(n,{frame_step_frames}))'"
        decoder_kwargs["-vf"] = vf_filter

    decoder = decoder_cls(video_path, frame_format="rgb24", **decoder_kwargs).formulate()

    processed = 0

    try:
        # FFmpeg select filter handles frame skipping, so we process all frames returned
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
                output_index = processed  # Use processed count as index

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
    """
    Preprocess video using NVDEC hardware acceleration via DeFFcode.
    Uses GPU-based decoding for maximum performance.
    Resizing is handled by PyTorch transforms for consistency with CPU preprocessing.
    """
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        raise RuntimeError("CUDA device unavailable for DeFFcode GPU preprocessing")

    mean, std = get_normalization_config(norm_config, device)

    # Always use PyTorch resize for consistency with CPU preprocessing
    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=True,
    )

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

    ffprefixes = ["-vsync", "0", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if device_index is not None:
        ffprefixes.extend(["-hwaccel_device", str(device_index)])

    frame_step_frames = 1
    if frame_interval and frame_interval > 0:
        if source_fps:
            frame_step_frames = max(1, round(source_fps * frame_interval))
        else:
            frame_step_frames = max(1, round(1.0 / frame_interval))

    vf_filters = []
    # Use frame-based select filter for consistency
    if frame_step_frames > 1:
        vf_filters.append(f"select=not(mod(n\\,{frame_step_frames}))")
    
    # Add hwdownload and format conversion for NVDEC
    vf_filters.extend(
        [
            "hwdownload",
            "format=nv12",
        ]
    )
    # Let DeFFcode handle the final nv12->rgb24 conversion (faster than in filter chain)
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

    processed = 0

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
        if frame_interval and frame_interval > 0:
            interval_fps = _target_fps(frame_interval)
            if interval_fps:
                effective_fps = interval_fps

        # FFmpeg select filter handles frame skipping, so we process all frames returned
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
    finally:
        terminate = getattr(decoder, "terminate", None)
        if callable(terminate):
            terminate()