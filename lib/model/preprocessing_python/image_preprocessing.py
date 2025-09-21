import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision
from lib.utils.torch_device_selector import get_preprocessing_device

decord.bridge.set_bridge('torch')

def get_normalization_config(index, device):
    normalization_configs = [
        (torch.tensor([0.485, 0.456, 0.406], device=device), torch.tensor([0.229, 0.224, 0.225], device=device)),
        (torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device), torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)),
    ]
    return normalization_configs[index]

def custom_round(x, base=1):
    return base * round(x/base)

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
    frameTransforms = None
    if vr_video:
        if (use_half_precision):
            frameTransforms = transforms.Compose([
                transforms.ToDtype(torch.float16, scale=True),
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            frameTransforms = transforms.Compose([
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                transforms.Normalize(mean=mean, std=std),
            ])
    else:
        if (use_half_precision):
            frameTransforms = transforms.Compose([
                transforms.ToDtype(torch.float16, scale=True),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            frameTransforms = transforms.Compose([
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=mean, std=std),
            ])
    return frameTransforms

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
    device = get_preprocessing_device(device)
    # MPS excluded - doesn't support BICUBIC interpolation
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
    

#TODO: TRY OTHER PREPROCESSING METHODS AND TRY MAKING PREPROCESSING TRUE ASYNC
def preprocess_video(video_path, frame_interval=0.5, img_size=512, use_half_precision=True, device=None, use_timestamps=False, vr_video=False, norm_config=1):
    device = get_preprocessing_device(device)
    # MPS excluded - doesn't support BICUBIC interpolation

    mean, std = get_normalization_config(norm_config, device)
    frame_transforms = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)
    vr = None
    if vr_video:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    else:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), width=img_size, height=img_size)
    fps = vr.get_avg_fps()
    frame_interval = custom_round(fps * frame_interval)
    for i in range(0, len(vr), frame_interval):
        frame = vr[i].to(device)
        if vr_video:
            frame = vr_permute(frame)
        frame = frame.permute(2, 0, 1)
        frame = frame_transforms(frame)
        frame_index = i
        if use_timestamps:
            frame_index = frame_index / custom_round(fps)
        yield (frame_index, frame)
    del vr
    
