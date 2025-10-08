import gc
import torch


def clear_gpu_cache():
    """
    Clear GPU cache for all available PyTorch backends.
    
    This function checks for CUDA, MPS, and XPU availability and clears
    their respective caches. Also runs garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
    
    # Run garbage collection to free up Python objects
    gc.collect()