import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision

# Make Decord give us torch tensors
decord.bridge.set_bridge('torch')


def get_normalization_config(index, device):
    normalization_configs = [
        (torch.tensor([0.485, 0.456, 0.406], device=device),
         torch.tensor([0.229, 0.224, 0.225], device=device)),
        (torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device),
         torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)),
    ]
    return normalization_configs[index]


def custom_round(x, base=1):
    return base * round(x / base)


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
    vr_video=False,
    img_size=512,
    need_resize=False
):
    dtype = torch.float16 if use_half_precision else torch.float32
    ops = [transforms.ToDtype(dtype, scale=True)]
    if need_resize:
        ops.insert(0, transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC))
    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def vr_permute(frame):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio > 1.5:
        # 180 VR: right half
        frame = frame[:, frame.shape[1] // 2:]
    else:
        # 360 VR: top half, center 50%
        frame = frame[:frame.shape[0] // 2, frame.shape[1] // 4:3 * frame.shape[1] // 4]
    return frame


def preprocess_image(image_path, img_size=512, use_half_precision=True, device=None, norm_config=1):
    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config, device)
    dtype = torch.float16 if use_half_precision else torch.float32
    image_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])
    return image_transforms(read_image(image_path).to(device))


def preprocess_video(
    video_path,
    frame_interval=0.5,
    img_size=512,
    use_half_precision=True,
    device=None,
    use_timestamps=False,
    vr_video=False,
    norm_config=1,
    decode_chunk_size=128
):
    """
    Yields (frame_index_or_timestamp, frame_tensor[C,H,W]).
    Efficient batched decode using decord.get_batch, optional NVDEC,
    and GPU transforms when device is CUDA.
    """
    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config, device)

    # Try NVDEC on CUDA; fallback CPU. Decoder-resize for non-VR.
    try:
        ctx = decord.gpu(0) if device.type == 'cuda' else decord.cpu(0)
        if vr_video:
            vr = decord.VideoReader(video_path, ctx=ctx)
            need_resize = True
        else:
            vr = decord.VideoReader(video_path, ctx=ctx, width=img_size, height=img_size)
            need_resize = False
    except Exception:
        ctx = decord.cpu(0)
        vr = decord.VideoReader(
            video_path, ctx=ctx,
            width=None if vr_video else img_size,
            height=None if vr_video else img_size
        )
        need_resize = vr_video or False

    fps = vr.get_avg_fps()
    step = max(int(custom_round(fps * frame_interval)), 1)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size, need_resize=need_resize)

    wanted = list(range(0, len(vr), step))

    def to_device_nb(t):
        if device.type == 'cuda':
            if t.device.type == 'cpu' and hasattr(t, "pin_memory"):
                try:
                    t = t.pin_memory()
                except RuntimeError:
                    pass
            return t.to(device, non_blocking=True)
        return t.to(device)

    for start in range(0, len(wanted), decode_chunk_size):
        ids = wanted[start:start + decode_chunk_size]
        # (T,H,W,C)
        frames_hwcn = vr.get_batch(ids)

        if vr_video:
            T, H, W, C = frames_hwcn.shape
            aspect_ratio = W / H
            if aspect_ratio > 1.5:
                frames_hwcn = frames_hwcn[:, :, W // 2:, :]
            else:
                frames_hwcn = frames_hwcn[:, :H // 2, W // 4:3 * W // 4, :]

        # -> (T,C,H,W)
        frames = frames_hwcn.permute(0, 3, 1, 2).contiguous()
        frames = to_device_nb(frames)
        frames = frame_transforms(frames)  # [T,C,H,W] on device

        for i, frame in enumerate(frames):
            idx = ids[i]
            if use_timestamps:
                idx = idx / max(int(custom_round(fps)), 1)
            yield (idx, frame)

    del vr
