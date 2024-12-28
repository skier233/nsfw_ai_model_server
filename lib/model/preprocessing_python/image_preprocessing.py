import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image
import torchvision

decord.bridge.set_bridge('torch')

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


def preprocess_image(image_path, img_size=512, use_half_precision=True, device=None):
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
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
#TODO: Make all preprocessing code open source and configurable
def preprocess_video_vr(video_path, frame_interval=0.5, img_size=512, use_half_precision=True, device=None, use_timestamps=False):
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
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
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    frame_interval = custom_round(fps * frame_interval)

    for i in range(0, len(vr), frame_interval):
        frame = vr[i].to(device)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        if aspect_ratio > 1.5:
            # 180 VR video, crop the right half of the image
            frame = frame[:, frame.shape[1]//2:]
        else:
            # 360 VR video, take the top half and the center 50% of the frame
            frame = frame[:frame.shape[0]//2, frame.shape[1]//4:3*frame.shape[1]//4]
        frame = frame.permute(2, 0, 1)
        frame = frameTransforms(frame)
        frame_index = i
        if use_timestamps:
            frame_index = frame_index / custom_round(fps)
        yield (frame_index, frame)
    del vr

def preprocess_video(video_path, frame_interval=0.5, img_size=512, use_half_precision=True, device=None, use_timestamps=False, vr_video=False):
    if vr_video:
        yield from preprocess_video_vr(video_path, frame_interval, img_size, use_half_precision, device, use_timestamps)
        return

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
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
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), width=img_size, height=img_size)
    fps = vr.get_avg_fps()
    frame_interval = custom_round(fps * frame_interval)

    for i in range(0, len(vr), frame_interval):
        frame = vr[i].to(device)
        frame = frame.permute(2, 0, 1)
        frame = frameTransforms(frame)
        frame_index = i
        if use_timestamps:
            frame_index = frame_index / custom_round(fps)
        yield (frame_index, frame)
    del vr