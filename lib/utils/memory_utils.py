import gc
import torch


def cgroup_memory_limit_bytes():
    """This process's cgroup memory limit in bytes, or None if unlimited/unknown.

    Handles cgroup v2 (``memory.max``) and v1 (``memory.limit_in_bytes``).  A
    literal ``"max"`` or the v1 near-2**63 "unlimited" sentinel is reported as
    None so callers fall back to host RAM.
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",                    # cgroup v2 (unified)
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            val = int(raw)
        except ValueError:
            continue
        if val <= 0:
            continue
        # cgroup v1 reports "unlimited" as a huge number (~2**63). Treat any
        # limit at/above 2**62 as effectively unlimited.
        if val >= (1 << 62):
            return None
        return val
    return None


def effective_total_memory_mb():
    """Total RAM available to *this* process in MB, honoring a container's cap.

    psutil reports the host's RAM even inside a memory-limited container, so a
    naive budget can target far more memory than the container is allowed to
    touch (and then OOM/livelock).  This clamps to the cgroup limit whenever one
    is set below host RAM.  Returns None only if neither source is available.
    """
    host_mb = None
    try:
        import psutil
        host_mb = psutil.virtual_memory().total / (1024 ** 2)
    except Exception:
        host_mb = None
    limit = cgroup_memory_limit_bytes()
    limit_mb = (limit / (1024 ** 2)) if limit else None
    vals = [v for v in (host_mb, limit_mb) if v and v > 0]
    if not vals:
        return None
    return min(vals)


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