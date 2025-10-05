import json

import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision
import numpy as np

from deffcode import FFdecoder as DeffcodeDecoder  # type: ignore[import]

decord.bridge.set_bridge('torch')

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
        tensor = torch.from_numpy(np.asarray(frame))

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
    return transforms.Compose([
        transforms.ToDtype(dtype, scale=True),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(mean=mean, std=std),
    ])

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
def preprocess_video(video_path, frame_interval=0.5, img_size=512, use_half_precision=True, device=None, use_timestamps=False, vr_video=False, norm_config=1):
    if device:
        device = torch.device(device)
    else:
        #Use CPU for Apple Silicon as well, because it cannot handle BICUBIC
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)
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

def preprocess_video_deffcode(video_path, frame_interval=0.5, img_size=512, use_half_precision=True, device=None, use_timestamps=False, vr_video=False, norm_config=1):

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)

    decoder_kwargs = {}
    if not vr_video and img_size:
        decoder_kwargs["-custom_resolution"] = (img_size, img_size)

    decoder = DeffcodeDecoder(video_path, frame_format="rgb24", **decoder_kwargs).formulate()

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