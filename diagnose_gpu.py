import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"XPU available: {torch.xpu.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device: {torch.cuda.device(0)}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA total memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"CUDA allocated memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"CUDA cached memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    print(f"CUDA free memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 3):.2f} GB")

if torch.xpu.is_available():
    print(f"XPU device count: {torch.xpu.device_count()}")
    print(f"XPU device: {torch.xpu.device(0)}")
    print(f"XPU device name: {torch.xpu.get_device_name(0)}")