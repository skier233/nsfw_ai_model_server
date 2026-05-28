import gc
import torch


def clear_gpu_cache():
    """
    Clear GPU cache for all available PyTorch backends.
    
    This function checks for CUDA, MPS, and XPU availability and clears
    their respective caches. Also runs garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        torch.xpu.empty_cache()

    gc.collect()