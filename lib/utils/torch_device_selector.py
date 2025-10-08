"""
PyTorch device selection utilities.
"""

import torch


def get_best_available_device():
    """
    Get the best available PyTorch device in order of preference:
    CUDA > XPU > MPS > CPU

    Returns:
        str: Device string ('cuda', 'xpu', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_torch_device(device=None):
    """
    Get a torch.device object, either from the provided device or auto-selected.

    Args:
        device (str or torch.device, optional): Specific device to use.
                                               If None, auto-selects best available.

    Returns:
        torch.device: PyTorch device object
    """
    if device:
        return torch.device(device)
    else:
        return torch.device(get_best_available_device())


def get_device_string(device=None):
    """
    Get device as a string, either from the provided device or auto-selected.

    Args:
        device (str or torch.device, optional): Specific device to use.
                                               If None, auto-selects best available.

    Returns:
        str: Device string ('cuda', 'xpu', 'mps', or 'cpu')
    """
    if device:
        if isinstance(device, torch.device):
            return device.type
        return str(device)
    else:
        return get_best_available_device()


def get_preprocessing_device(device=None):
    """
    Get the best available PyTorch device for preprocessing operations.
    Excludes MPS as it doesn't support BICUBIC interpolation used in image preprocessing.
    Order of preference: CUDA > XPU > CPU (MPS is never selected)

    Args:
        device (str or torch.device, optional): Specific device to use.
                                               If None, auto-selects best available.

    Returns:
        torch.device: PyTorch device object (never MPS)
    """
    if device:
        return torch.device(device)

    # Auto-select best device, excluding MPS
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")
