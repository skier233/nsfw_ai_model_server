import io
import json
import logging
import os
import queue
import re
import threading
import zipfile
from contextlib import suppress
from typing import IO, Optional

import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor
from torchvision.io import decode_image, ImageReadMode
import torchvision
import numpy as np
#from deffcode import Sourcer

from lib.pipeline.preprocess_spec import NORMALIZATION_PRESETS


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


_ZIP_PATH_PATTERN = re.compile(r"\.zip(?=$|[\\/])", re.IGNORECASE)

_URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _normalize_video_source(video_path: str) -> str:
    if not isinstance(video_path, str):
        return video_path

    candidate = video_path.strip()
    if not candidate:
        return candidate

    # On Linux containers, clients sometimes send Windows-style paths.
    # Only rewrite if it results in an existing path to stay conservative.
    if os.name != "nt" and "\\" in candidate:
        rewritten = candidate.replace("\\", "/")
        if os.path.exists(rewritten):
            return rewritten

    return candidate


def _validate_local_video_source(video_path: str) -> str:
    normalized = _normalize_video_source(video_path)

    if not normalized or not isinstance(normalized, str):
        raise FileNotFoundError("Video path is empty or invalid")

    # Allow URL-like sources (rtsp/http/etc) without local existence checks.
    if _URL_SCHEME_PATTERN.match(normalized):
        return normalized

    if not os.path.exists(normalized):
        raise FileNotFoundError(
            f"Video file not found: '{normalized}'. "
            "If you're running the server in Docker, make sure the host folder containing the video is mounted into the container, "
            "and path mapping configured in AI Overhaul"
        )

    if not os.path.isfile(normalized):
        raise FileNotFoundError(f"Video path is not a file: '{normalized}'")

    return normalized


def _is_rocm_build() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _split_zip_archive_path(path: str) -> Optional[tuple[str, str]]:
    match = _ZIP_PATH_PATTERN.search(path)
    if not match:
        return None
    archive_path = path[: match.end()]
    inner_path = path[match.end() :].lstrip("\\/")
    if not inner_path:
        raise DimensionError(
            f"Invalid Image Path: '{path}' points to a zip archive but no inner file was specified."
        )
    return archive_path, inner_path


def _normalized_zip_member_path(member_path: str) -> str:
    candidate = member_path.replace("\\", "/")
    return candidate.lstrip("/")


def _load_image_from_zip(archive_path: str, member_path: str) -> torch.Tensor:
    normalized_member = _normalized_zip_member_path(member_path)
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Zip archive '{archive_path}' does not exist.")

    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(normalized_member) as member_file:
                payload = member_file.read()
        except KeyError as exc:
            raise FileNotFoundError(
                f"File '{normalized_member}' not found inside archive '{archive_path}'."
            ) from exc

    stream = io.BytesIO(payload)
    pseudo_path = f"{archive_path}!{normalized_member}"

    if _is_rocm_build():
        return _load_image_with_pillow(pseudo_path, file_obj=stream)

    byte_tensor = torch.frombuffer(memoryview(payload), dtype=torch.uint8).clone()
    decoded = decode_image(byte_tensor, mode=ImageReadMode.RGB)
    return _finalize_decoded_tensor(decoded, pseudo_path)


def _load_image_with_pillow(image_path: str, *, file_obj: Optional[IO[bytes]] = None) -> torch.Tensor:
    from PIL import Image

    if file_obj is not None:
        file_obj.seek(0)
        image_source = file_obj
    else:
        image_source = image_path

    with Image.open(image_source) as img:
        if getattr(img, "is_animated", False):
            frame_count = getattr(img, "n_frames", 1)
            middle_index = max(frame_count // 2, 0)
            img.seek(middle_index)
        frame = img.convert("RGB")
        tensor = pil_to_tensor(frame)
    return _ensure_rgb_channels(tensor, image_path)


def _load_image_tensor(image_path: str) -> torch.Tensor:
    archive_parts = _split_zip_archive_path(image_path)
    if archive_parts is not None:
        archive_path, member_path = archive_parts
        return _load_image_from_zip(archive_path, member_path)

    if _is_rocm_build():
        return _load_image_with_pillow(image_path)

    tensor = decode_image(image_path, mode=ImageReadMode.RGB)
    return _finalize_decoded_tensor(tensor, image_path)


def _ensure_rgb_channels(tensor: torch.Tensor, image_path: str) -> torch.Tensor:
    if tensor.ndim < 3:
        raise DimensionError(
            f"Invalid Image: Unable to decode '{image_path}' into a 3-channel tensor; got shape {tuple(tensor.shape)}."
        )

    channel_count = tensor.shape[0]
    if channel_count == 3:
        return tensor

    if channel_count == 1:
        return tensor.repeat(3, 1, 1)

    if channel_count >= 4:
        return tensor[:3]

    raise DimensionError(
        f"Invalid Image: Unsupported channel count ({channel_count}) for '{image_path}'."
    )


def _finalize_decoded_tensor(tensor: torch.Tensor, image_path: str) -> torch.Tensor:
    if tensor.ndim == 4:
        frame_count = tensor.shape[0]
        middle_index = frame_count // 2
        tensor = tensor[middle_index]
    return _ensure_rgb_channels(tensor, image_path)

def get_normalization_config(index, device):
    if index == -1:
        return None, None
    mean_list, std_list = NORMALIZATION_PRESETS[index]
    return (
        torch.tensor(mean_list, device=device),
        torch.tensor(std_list, device=device),
    )

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
    video_path = _validate_local_video_source(video_path)
    video = torchvision.io.VideoReader(video_path, "video")
    metadata = video.get_metadata()
    duration = metadata['video']['duration'][0]
    return duration

def get_video_duration_av(video_path):
    import av
    video_path = _validate_local_video_source(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return container.duration / av.time_base
        # Fall back to frame-count / frame-rate
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames = stream.frames
        return (frames / fps) if fps and frames else 0.0

def get_video_duration_deffcode(video_path):

    video_path = _validate_local_video_source(video_path)
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


def get_frame_transforms(
    use_half_precision,
    mean,
    std,
    vr_video: bool = False,
    img_size: int = 512,
    apply_resize: bool = True,
    scale_values: bool = True,
):
    dtype = torch.float16 if use_half_precision else torch.float32
    transforms_list = [
        transforms.ToDtype(dtype, scale=scale_values),
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

def _prepare_frame(frame, device, vr_video, frame_transforms):
    # Only pin memory when targeting a CUDA device — pinning for a
    # CPU-only path wastes time and can cause CUDA synchronisation.
    _pin = device is not None and device.type != "cpu"
    tensor = _ensure_torch_tensor(frame, device, pin_memory=_pin, non_blocking=_pin)
    if vr_video:
        tensor = vr_permute(tensor)
    tensor = tensor.permute(2, 0, 1)
    return frame_transforms(tensor)


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
    max_decode_long_edge=0,
    gpu_min_long_edge=30600,
):
    """
    Automatically select GPU (NVDEC) or CPU preprocessing based on video resolution.

    When the video's longest edge is >= *gpu_min_long_edge* AND a CUDA device is
    available, DeFFcode GPU (NVDEC) decoding is used — it provides the biggest
    win at high resolutions where decode bandwidth is the bottleneck.  For
    lower-resolution videos, CPU decoding avoids contending with AI inference
    for GPU resources.

    *gpu_min_long_edge* defaults to 3600 (just below 4K) but can be tuned
    per-pipeline via the ``gpu_min_long_edge`` config key in the preprocessor
    model YAML.  Set to 0 to always use GPU, or a very large value to always
    use CPU.
    """
    video_path = _validate_local_video_source(video_path)
    if device:
        target_device = torch.device(device)
    else:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_gpu_preprocessing = False
    min_long_edge = int(gpu_min_long_edge) if gpu_min_long_edge else 0

    # Determine if GPU preprocessing should be used based on resolution
    if target_device.type == 'cuda':
        if min_long_edge <= 0:
            # 0 means always GPU
            use_gpu_preprocessing = True
            _LOGGER.debug("gpu_min_long_edge=0: always using GPU preprocessing")
        else:
            try:
                sourcer = Sourcer(video_path).probe_stream()
                try:
                    metadata = sourcer.retrieve_metadata()
                    width = metadata.get("source_video_resolution", [0, 0])[0]
                    height = metadata.get("source_video_resolution", [0, 0])[1]
                    long_edge = max(width, height)
                    if long_edge >= min_long_edge:
                        use_gpu_preprocessing = True
                        _LOGGER.debug(
                            "Video resolution %dx%d (long edge %d >= %d): using GPU preprocessing",
                            width, height, long_edge, min_long_edge,
                        )
                    else:
                        _LOGGER.debug(
                            "Video resolution %dx%d (long edge %d < %d): using CPU preprocessing",
                            width, height, long_edge, min_long_edge,
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
            yielded_any = False
            for item in preprocess_video_deffcode_gpu(
                video_path=video_path,
                frame_interval=frame_interval,
                img_size=img_size,
                use_half_precision=use_half_precision,
                device=target_device,
                use_timestamps=use_timestamps,
                vr_video=vr_video,
                norm_config=norm_config,
                max_decode_long_edge=max_decode_long_edge,
            ):
                yielded_any = True
                yield item
            if yielded_any:
                return
            else:
                _LOGGER.warning(
                    "GPU preprocessing yielded no frames for '%s'. Falling back to CPU.",
                    video_path,
                )
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
        max_decode_long_edge=max_decode_long_edge,
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
    max_decode_long_edge=0,
):
    """
    Preprocess video using CPU-based DeFFcode with FFmpeg filtering.
    Uses FFmpeg select filter for efficient frame skipping.
    """
    video_path = _validate_local_video_source(video_path)
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
        apply_resize=(isinstance(img_size, (int, float)) and int(img_size) > 0),
        scale_values=(norm_config != -1),
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
    
    vf_parts = []
    if frame_step_frames > 1:
        vf_parts.append(f"select='not(mod(n,{frame_step_frames}))'")

    # Downscale at FFmpeg level when max_decode_long_edge is set.
    # This dramatically reduces per-frame data size before it reaches Python.
    # Uses FFmpeg expressions: scale only if the source exceeds the cap.
    _mdle = int(max_decode_long_edge) if max_decode_long_edge else 0
    if _mdle > 0:
        # scale='if(gt(max(iw,ih),CAP), if(gt(iw,ih), CAP, -2), iw)' : ...
        # More readable: if long_edge > cap, scale down preserving aspect ratio.
        vf_parts.append(
            f"scale='if(gt(max(iw\\,ih)\\,{_mdle})\\,if(gt(iw\\,ih)\\,{_mdle}\\,-2)\\,iw)'"
            f":'if(gt(max(iw\\,ih)\\,{_mdle})\\,if(gt(ih\\,iw)\\,{_mdle}\\,-2)\\,ih)'"
            ":flags=bilinear"
        )

    if vf_parts:
        decoder_kwargs["-vf"] = ",".join(vf_parts)

    decoder = decoder_cls(video_path, frame_format="rgb24", **decoder_kwargs).formulate()

    processed = 0

    try:
        # FFmpeg select filter handles frame skipping, so we process all frames returned
        for index, frame in enumerate(decoder.generateFrame()):
            if frame is None:
                continue

            result = _prepare_frame(frame, device, vr_video, frame_transforms)

            if use_timestamps:
                if frame_interval and frame_interval > 0:
                    output_index = processed * frame_interval
                else:
                    output_index = index / (effective_fps or 1.0)
            else:
                output_index = processed  # Use processed count as index

            yield (output_index, result)
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
    max_decode_long_edge=0,
):
    """
    Preprocess video using NVDEC hardware acceleration via DeFFcode.
    Uses GPU-based decoding for maximum performance.
    Resizing is handled by PyTorch transforms for consistency with CPU preprocessing.
    """
    video_path = _validate_local_video_source(video_path)
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
        apply_resize=(isinstance(img_size, (int, float)) and int(img_size) > 0),
        scale_values=(norm_config != -1),
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

    # Downscale at FFmpeg level after hwdownload when max_decode_long_edge
    # is set.  This reduces the per-frame data size before Python sees it.
    _mdle = int(max_decode_long_edge) if max_decode_long_edge else 0
    if _mdle > 0:
        vf_filters.append(
            f"scale='if(gt(max(iw\\,ih)\\,{_mdle})\\,if(gt(iw\\,ih)\\,{_mdle}\\,-2)\\,iw)'"
            f":'if(gt(max(iw\\,ih)\\,{_mdle})\\,if(gt(ih\\,iw)\\,{_mdle}\\,-2)\\,ih)'"
            ":flags=bilinear"
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

            result = _prepare_frame(frame, device, vr_video, frame_transforms)

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

            yield (output_index, result)
            processed += 1
    finally:
        terminate = getattr(decoder, "terminate", None)
        if callable(terminate):
            terminate()


# ---------------------------------------------------------------------------
# PyAV backends
# ---------------------------------------------------------------------------

def _av_resize_frame_if_needed(frame_np, max_long_edge: int):
    """Downscale a numpy HWC frame so its longest edge <= *max_long_edge*.

    Uses a two-pass strategy for large downscale ratios (> 2×):
      1. Coarse subsample via numpy stride slicing  (near-zero cost)
      2. Fine resize via Pillow/cv2 from the smaller intermediate

    For 4K→512 this is ~4-5× faster than a single Pillow BILINEAR pass.
    """
    if max_long_edge <= 0:
        return frame_np
    h, w = frame_np.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return frame_np
    scale = max_long_edge / long_edge
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    # ---- coarse stride subsample when downscaling > 2× ----
    stride = max(1, int(long_edge / (max_long_edge * 2)))
    if stride > 1:
        frame_np = frame_np[::stride, ::stride, :].copy()

    try:
        import cv2
        return cv2.resize(frame_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        import PIL.Image
        img = PIL.Image.fromarray(frame_np)
        img = img.resize((new_w, new_h), PIL.Image.BILINEAR)
        return np.asarray(img)


def preprocess_video_av(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
    max_decode_long_edge=0,
    **_kwargs,
):
    """Preprocess video using PyAV with multi-threaded decoding.

    Decodes every frame and skips in Python (like the DeFFcode CPU backend).
    ``stream.thread_type = "AUTO"`` lets libav use frame- or slice-level
    threading for the codec, which is competitive with DeFFcode on 1080p
    and faster on 4K content.
    """
    import av

    video_path = _validate_local_video_source(video_path)
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=(isinstance(img_size, (int, float)) and int(img_size) > 0),
        scale_values=(norm_config != -1),
    )

    _mdle = int(max_decode_long_edge) if max_decode_long_edge else 0

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        source_fps = float(stream.average_rate) if stream.average_rate else 30.0
        frame_step = max(1, round(source_fps * frame_interval)) if frame_interval and frame_interval > 0 else 1

        processed = 0
        for index, frame in enumerate(container.decode(video=0)):
            if frame_step > 1 and (index % frame_step) != 0:
                continue

            frame_np = frame.to_ndarray(format="rgb24")
            if _mdle > 0:
                frame_np = _av_resize_frame_if_needed(frame_np, _mdle)

            result = _prepare_frame(frame_np, device, vr_video, frame_transforms)

            if use_timestamps:
                output_index = processed * frame_interval if (frame_interval and frame_interval > 0) else index / source_fps
            else:
                output_index = processed

            yield (output_index, result)
            processed += 1
    finally:
        container.close()


def preprocess_video_av_seek(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
    max_decode_long_edge=0,
    **_kwargs,
):
    """Preprocess video using PyAV with seek-based frame extraction.

    Instead of decoding every frame, this seeks to each target timestamp
    and decodes only one frame per seek.  Much faster than full-decode
    when ``frame_interval`` is large (>= 1 s) or the video is long,
    because it skips most of the bitstream entirely.

    A background prefetch thread runs the seek loop ahead of the consumer
    so that seek I/O overlaps with downstream GPU inference.  PyAV's
    C-level seek/decode releases the GIL, making true concurrency possible.
    """
    import av

    video_path = _validate_local_video_source(video_path)
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=(isinstance(img_size, (int, float)) and int(img_size) > 0),
        scale_values=(norm_config != -1),
    )

    _mdle = int(max_decode_long_edge) if max_decode_long_edge else 0

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Determine duration in seconds
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = 0.0

        if not frame_interval or frame_interval <= 0:
            frame_interval = 1.0 / (float(stream.average_rate) if stream.average_rate else 30.0)

        time_base = stream.time_base

        # ---- prefetch machinery ----
        # The background thread runs ONLY the GIL-free C-level work:
        # seek, decode, to_ndarray, and optional cv2 resize.  All
        # Python/torch work (_prepare_frame) stays on the consumer side
        # so it doesn't compete with model inference for the GIL.
        _PREFETCH_DEPTH = 16
        _SENTINEL = None  # signals end-of-stream or error
        prefetch_q: queue.Queue = queue.Queue(maxsize=_PREFETCH_DEPTH)
        prefetch_error: list = []  # mutable container for thread exception

        def _prefetch_worker():
            """Seek-decode loop that feeds *prefetch_q* with raw numpy frames."""
            try:
                idx = 0
                t = 0.0
                while t < duration:
                    target_pts = int(t / time_base)
                    container.seek(target_pts, stream=stream)

                    frame = None
                    for f in container.decode(video=0):
                        frame = f
                        break
                    if frame is None:
                        break

                    # All C-level / GIL-free work:
                    frame_np = frame.to_ndarray(format="rgb24")
                    if _mdle > 0:
                        frame_np = _av_resize_frame_if_needed(frame_np, _mdle)

                    if use_timestamps:
                        out_idx = t
                    else:
                        out_idx = idx

                    prefetch_q.put((out_idx, frame_np))  # blocks if queue is full
                    idx += 1
                    t += frame_interval
            except Exception as exc:
                prefetch_error.append(exc)
            finally:
                prefetch_q.put(_SENTINEL)

        worker = threading.Thread(target=_prefetch_worker, daemon=True)
        worker.start()

        try:
            while True:
                item = prefetch_q.get()
                if item is _SENTINEL:
                    if prefetch_error:
                        raise prefetch_error[0]
                    return
                out_idx, frame_np = item
                # Torch work runs on consumer thread — no GIL contention
                # with the prefetch thread's C-level seeks.
                result = _prepare_frame(frame_np, device, vr_video, frame_transforms)
                yield (out_idx, result)
        finally:
            # Drain the queue so the worker thread isn't blocked on put()
            # and can terminate cleanly.
            while worker.is_alive():
                try:
                    prefetch_q.get_nowait()
                except queue.Empty:
                    break
            worker.join(timeout=2.0)
    finally:
        container.close()