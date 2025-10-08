import json
import logging
import os
import queue
import re
import threading
from contextlib import suppress
from time import perf_counter
from typing import Optional

import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision
import numpy as np

try:
    from torchaudio.io import StreamReader as TorchaudioStreamReader

    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover - optional dependency
    TorchaudioStreamReader = None
    _HAS_TORCHAUDIO = False

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
        #array.setflags(write=True)
        tensor = torch.from_numpy(array)

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

def get_video_duration_decord(video_path):
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
    

def _prepare_frame(frame, device, vr_video, frame_transforms, *, profile_stats=None):
    pin_for_gpu = False
    non_blocking = False
    timing_enabled = profile_stats is not None
    prepare_start = perf_counter() if timing_enabled else None
    ensure_start = perf_counter() if timing_enabled else None

    tensor = _ensure_torch_tensor(
        frame,
        device,
        pin_memory=pin_for_gpu,
        non_blocking=non_blocking,
    )

    if timing_enabled:
        now = perf_counter()
        profile_stats.setdefault("ensure_tensor_time", 0.0)
        profile_stats["ensure_tensor_time"] += now - ensure_start
        segment_start = now
    else:
        segment_start = None

    if vr_video:
        tensor = vr_permute(tensor)
        if timing_enabled:
            new_now = perf_counter()
            profile_stats.setdefault("vr_permute_time", 0.0)
            profile_stats["vr_permute_time"] += new_now - segment_start
            segment_start = new_now

    tensor = tensor.permute(2, 0, 1)

    if timing_enabled:
        new_now = perf_counter()
        profile_stats.setdefault("permute_time", 0.0)
        profile_stats["permute_time"] += new_now - segment_start
        segment_start = new_now

    tensor = frame_transforms(tensor)

    if timing_enabled:
        if tensor.is_cuda:
            torch.cuda.synchronize(device)
        new_now = perf_counter()
        profile_stats.setdefault("transforms_time", 0.0)
        profile_stats["transforms_time"] += new_now - segment_start
        profile_stats.setdefault("prepare_time", 0.0)
        profile_stats["prepare_time"] += new_now - prepare_start

    return tensor


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

    use_gpu_resize = (
        target_height is not None
        and target_width is not None
        and target_height > 0
        and target_width > 0
        and not vr_video
    )

    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=not use_gpu_resize,
    )
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

    use_gpu_resize = (
        target_height is not None
        and target_width is not None
        and target_height > 0
        and target_width > 0
        and not vr_video
    )

    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=not use_gpu_resize,
    )

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


def preprocess_video_deffcode_gpu_minimal(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
):
    """Minimal fast implementation based on working experiment."""
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        raise RuntimeError("CUDA device unavailable")

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=False,  # GPU resize in FFmpeg
    )

    # Minimal decoder setup like the fast experiment
    base_prefixes = ['-vsync', '0', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    if device.index is not None:
        base_prefixes.extend(['-hwaccel_device', str(device.index)])
    elif torch.cuda.device_count() > 0:
        base_prefixes.extend(['-hwaccel_device', str(torch.cuda.current_device())])

    interval_expr = _format_ffmpeg_float(frame_interval) if frame_interval and frame_interval > 0 else "2"
    vf = f"select='not(mod(t,{interval_expr}))',scale_cuda={img_size}:{img_size}:interp_algo=bicubic,hwdownload,format=nv12"

    kwargs = {
        '-vcodec': 'hevc_cuvid',
        '-ffprefixes': base_prefixes,
        '-vf': vf,
    }

    decoder_cls = _get_deffcode_decoder()
    decoder = decoder_cls(video_path, frame_format='rgb24', **kwargs).formulate()

    processed = 0
    try:
        for frame in decoder.generateFrame():
            if frame is None:
                continue

            # Simple conversion like the experiment
            array = np.asarray(frame).copy()  # Make writable copy
            tensor = torch.from_numpy(array)
            tensor = tensor.permute(2, 0, 1).to(device)
            tensor = frame_transforms(tensor)

            if use_timestamps:
                output_index = processed * frame_interval if frame_interval and frame_interval > 0 else float(processed)
            else:
                output_index = processed

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
    _disable_gpu_resize: bool = False,
    profile: Optional[dict] = None,
):

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        raise RuntimeError("CUDA device unavailable for DeFFcode GPU preprocessing")

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

    use_gpu_resize = (
        not _disable_gpu_resize
        and target_height is not None
        and target_width is not None
        and target_height > 0
        and target_width > 0
        and not vr_video
    )

    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
        apply_resize=not use_gpu_resize,
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
    if frame_interval and frame_interval > 0:
        interval_expr = _format_ffmpeg_float(frame_interval)
        vf_filters.append(f"select='not(mod(t,{interval_expr}))'")
    elif frame_step_frames > 1:
        vf_filters.append(f"select=not(mod(n\\,{frame_step_frames}))")
    if use_gpu_resize:
        interpolation = "bicubic"
        vf_filters.append(
            f"scale_cuda={target_width}:{target_height}:interp_algo={interpolation}"
        )
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

    profile_enabled = profile is not None
    if profile_enabled:
        profile.setdefault("frames", 0)
        profile.setdefault("decoded_frames", 0)
        profile.setdefault("skipped_frames", 0)
        profile.setdefault("decode_time", 0.0)
        profile.setdefault("ensure_tensor_time", 0.0)
        profile.setdefault("vr_permute_time", 0.0)
        profile.setdefault("permute_time", 0.0)
        profile.setdefault("transforms_time", 0.0)
        profile.setdefault("prepare_time", 0.0)
        profile.setdefault("total_time", 0.0)
        profile.setdefault("emitted_indices", [])

    processed = 0
    wall_start = perf_counter() if profile_enabled else None

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

        frame_iter = decoder.generateFrame()
        index = 0
        while True:
            decode_start = perf_counter() if profile_enabled else None
            try:
                frame = next(frame_iter)
            except StopIteration:
                break

            if profile_enabled:
                profile["decode_time"] += perf_counter() - decode_start
                profile["decoded_frames"] += 1

            current_index = index
            index += 1

            if frame is None:
                continue

            # FFmpeg select filter already handles frame skipping, no need to skip in Python
            tensor = _prepare_frame(
                frame,
                device,
                vr_video,
                frame_transforms,
                profile_stats=profile if profile_enabled else None,
            )

            if use_timestamps:
                if frame_interval and frame_interval > 0:
                    output_index = processed * frame_interval
                else:
                    output_index = current_index / (effective_fps or 1.0)
            else:
                if frame_interval and frame_interval > 0:
                    output_index = processed * frame_step_frames
                else:
                    output_index = current_index

            yield (output_index, tensor)
            processed += 1
            if profile_enabled:
                profile["frames"] += 1
                profile["emitted_indices"].append(current_index)

        if processed == 0:
            if use_gpu_resize and not _disable_gpu_resize:
                _LOGGER.warning(
                    "DeFFcode CUDA pipeline produced zero frames for '%s' when using scale_cuda; retrying without GPU resize.",
                    video_path,
                )
            else:
                _LOGGER.warning(
                    "DeFFcode CUDA pipeline produced zero frames for '%s'; falling back to CPU backend.",
                    video_path,
                )
    finally:
        terminate = getattr(decoder, "terminate", None)
        if callable(terminate):
            terminate()
        if profile_enabled and wall_start is not None:
            profile["total_time"] += perf_counter() - wall_start
            _LOGGER.info(
                "DeFFcode CUDA profile: frames=%d decoded=%d skipped=%d total=%.3fs decode=%.3fs prepare=%.3fs ensure=%.3fs permute=%.3fs transforms=%.3fs",
                profile.get("frames", 0),
                profile.get("decoded_frames", 0),
                profile.get("skipped_frames", 0),
                profile.get("total_time", 0.0),
                profile.get("decode_time", 0.0),
                profile.get("prepare_time", 0.0),
                profile.get("ensure_tensor_time", 0.0),
                profile.get("permute_time", 0.0),
                profile.get("transforms_time", 0.0),
            )

    if processed == 0:
        if use_gpu_resize and not _disable_gpu_resize:
            yield from preprocess_video_deffcode_gpu(
                video_path,
                frame_interval=frame_interval,
                img_size=img_size,
                use_half_precision=use_half_precision,
                device=device,
                use_timestamps=use_timestamps,
                vr_video=vr_video,
                norm_config=norm_config,
                _disable_gpu_resize=True,
                profile=profile,
            )
            return

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


def preprocess_video_torchaudio_gpu(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
):
    if not _HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio is not installed; install torchaudio with CUDA support to use this backend")

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        raise RuntimeError("CUDA device unavailable for torchaudio GPU preprocessing")

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(
        use_half_precision,
        mean,
        std,
        vr_video=vr_video,
        img_size=img_size,
    )

    # Mitigate duplicate OpenMP runtime errors on Windows when torchaudio initialises CUDA
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE"))

    try:
        metadata_probe_stream = TorchaudioStreamReader(video_path)
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"Failed to open '{video_path}' via torchaudio StreamReader") from exc

    try:
        src_info = metadata_probe_stream.get_src_stream_info(0)
    except Exception:
        src_info = None
    finally:
        metadata_probe_stream = None

    codec_name = None
    src_frame_rate = None
    if src_info is not None:
        codec_name = getattr(src_info, "codec_name", None) or getattr(src_info, "codec", None)
        src_frame_rate = (
            getattr(src_info, "frame_rate", None)
            or getattr(src_info, "avg_frame_rate", None)
            or getattr(src_info, "frame_rate_num", None)
        )

    decoder_name = _guess_deffcode_hw_decoder(codec_name)
    if decoder_name is None:
        raise RuntimeError(
            "Unable to determine a torchaudio CUDA decoder for codec "
            f"'{codec_name}'. Set DEFFCODE_HW_DECODER or re-encode the source."
        )

    if device.index is not None:
        hw_accel_value = f"cuda:{device.index}"
    else:
        hw_accel_value = "cuda"

    target_fps = _target_fps(frame_interval) if frame_interval and frame_interval > 0 else None
    frame_rate_override: Optional[float] = None
    if target_fps:
        # Always use FFmpeg fps filter when we have a target FPS to avoid decoding unnecessary frames
        frame_rate_override = target_fps

    frames_per_chunk = 1

    add_kwargs = {
        "frames_per_chunk": frames_per_chunk,
        "buffer_chunk_size": 3,
        "decoder": decoder_name,
        "hw_accel": hw_accel_value,
        "format": None,
    }
    if frame_rate_override is not None:
        add_kwargs["frame_rate"] = frame_rate_override

    primary_kwargs = dict(add_kwargs)

    try_configs = [("cuda", primary_kwargs)]

    if primary_kwargs.get("buffer_chunk_size") is not None:
        trimmed_kwargs = dict(primary_kwargs)
        trimmed_kwargs.pop("buffer_chunk_size", None)
        try_configs.append(("cuda-trimmed", trimmed_kwargs))

    cpu_kwargs = {
        "frames_per_chunk": primary_kwargs.get("frames_per_chunk", frames_per_chunk),
        "buffer_chunk_size": primary_kwargs.get("buffer_chunk_size"),
        "decoder": None,
        "hw_accel": None,
        "format": "rgb24",
    }
    if "frame_rate" in primary_kwargs:
        cpu_kwargs["frame_rate"] = primary_kwargs["frame_rate"]
    try_configs.append(("cpu", cpu_kwargs))

    def _build_stream():
        return TorchaudioStreamReader(video_path)

    configured_stream = None
    last_exc: Optional[Exception] = None

    def _call_kwargs(mapping: dict) -> dict:
        call = {}
        for key, value in mapping.items():
            if value is None and key not in {"format"}:
                continue
            call[key] = value
        return call

    for idx, (label, kwargs) in enumerate(try_configs):
        stream_candidate = _build_stream()
        add_method_candidate = getattr(stream_candidate, "add_basic_video_stream", None)
        method_label = "add_basic_video_stream"
        if add_method_candidate is None:
            add_method_candidate = getattr(stream_candidate, "add_video_stream", None)
            method_label = "add_video_stream"
            if add_method_candidate is None:
                last_exc = RuntimeError("torchaudio StreamReader lacks video stream helpers")
                _LOGGER.debug("torchaudio stream missing video stream helpers during '%s'", label)
                continue

        _LOGGER.debug(
            "torchaudio attempt '%s': decoder=%s hw_accel=%s frames_per_chunk=%s buffer_chunk_size=%s frame_rate=%s method=%s",
            label,
            kwargs.get("decoder"),
            kwargs.get("hw_accel"),
            kwargs.get("frames_per_chunk"),
            kwargs.get("buffer_chunk_size"),
            kwargs.get("frame_rate"),
            method_label,
        )
        call_kwargs = _call_kwargs(kwargs)
        _LOGGER.debug("torchaudio kwargs for '%s': %s", label, call_kwargs)

        try:
            add_method_candidate(**call_kwargs)
            configured_stream = stream_candidate
            if label == "cpu":
                _LOGGER.warning(
                    "torchaudio GPU decoding unavailable for '%s' (codec=%s); using software fallback.",
                    video_path,
                    codec_name,
                )
            break
        except Exception as exc:
            last_exc = exc
            _LOGGER.debug(
                "torchaudio add_video_stream config '%s' failed: %s",
                label,
                exc,
                exc_info=True,
            )
            continue

    if configured_stream is None:
        raise RuntimeError("Failed to configure torchaudio CUDA decoding stream") from last_exc

    stream = configured_stream

    if frame_rate_override is not None:
        effective_fps = float(frame_rate_override)
    else:
        effective_fps = _parse_fps_value(src_frame_rate) if src_frame_rate is not None else 30.0

    stride = 1
    if frame_interval and frame_interval > 0 and not frame_rate_override and effective_fps:
        stride = max(1, round(effective_fps * frame_interval))
    if stride <= 0:
        stride = 1

    total_index = 0
    processed = 0

    for chunk in stream.stream(timeout=0.0):
        if not chunk:
            continue
        video_batch = chunk[0]
        if video_batch is None:
            continue
        if isinstance(video_batch, np.ndarray):
            video_batch = torch.from_numpy(video_batch)
        if not isinstance(video_batch, torch.Tensor):
            continue
        if video_batch.ndim == 3:
            video_batch = video_batch.unsqueeze(0)
        if video_batch.numel() == 0:
            continue

        for frame in video_batch:
            if stride > 1 and (total_index % stride) != 0:
                total_index += 1
                continue

            current_index = total_index
            total_index += 1

            tensor = frame
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            if tensor.ndim != 3:
                raise RuntimeError(f"Unexpected torchaudio frame shape: {tuple(tensor.shape)}")

            if tensor.shape[0] == 1:
                tensor = tensor.expand(3, -1, -1)

            if vr_video:
                tensor_hwc = tensor.permute(1, 2, 0)
                tensor_hwc = vr_permute(tensor_hwc)
                tensor = tensor_hwc.permute(2, 0, 1)

            tensor = tensor.contiguous()
            if tensor.device != device:
                tensor = tensor.to(device, non_blocking=True)
            tensor = frame_transforms(tensor)

            if use_timestamps:
                if frame_interval and frame_interval > 0:
                    if effective_fps:
                        timestamp = current_index / effective_fps
                    else:
                        timestamp = processed * frame_interval
                else:
                    timestamp = current_index / effective_fps if effective_fps else float(current_index)
            else:
                timestamp = processed

            yield (timestamp, tensor)
            processed += 1

    if processed == 0:
        raise RuntimeError(
            "torchaudio CUDA pipeline produced zero frames. Check that the FFmpeg shared libraries and CUDA decoder are available."
        )