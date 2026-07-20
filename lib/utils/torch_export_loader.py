"""
Device retargeting for torch.export (.pt2) programs.

Models are exported on CPU so the archive is device-neutral: a CUDA-traced
archive records ``device: cuda`` in its weight metadata and torch.export.load
materializes the weights onto that device *during load*, which raises
"Torch not compiled with CUDA enabled" on a CPU-only or XPU build before any
fixup could run.

A CPU-traced archive loads anywhere, but it is not automatically portable
either: torch.export bakes the tracing device into the graph in two forms,
and moving a program between devices requires fixing both.

  1. ``torch.device`` objects -- weights, lifted constants, and factory-op
     kwargs (``aten.arange``, ``aten._assert_tensor_metadata``).  Handled by
     ``move_to_device_pass``.
  2. Bare device *strings* -- the ``wrap_with_autocast`` higher-order op stores
     its device as ``"cpu"`` / ``"cuda"``, not a ``torch.device``, so
     ``move_to_device_pass`` leaves it untouched.  On an FP16 model this is not
     cosmetic: the autocast region silently stops applying on the new device,
     the graph's explicit casts never happen, and inference fails with
     "mat1 and mat2 must have the same dtype, Float and Half".

``module().to(device)`` alone fixes neither -- it moves parameters and buffers
but not lifted constants or graph literals.
"""

import torch

try:
    from torch.export.passes import move_to_device_pass
except ImportError:  # torch < 2.6
    move_to_device_pass = None


_DEVICE_STRINGS = ("cpu", "cuda", "xpu", "mps", "hpu")


def _iter_graphs(exported_program):
    """Yield every (module, graph) pair, including subgraphs.

    Autocast regions and other higher-order ops live in submodules, so a scan
    of the top-level graph alone misses them.
    """
    for _, submodule in exported_program.graph_module.named_modules():
        graph = getattr(submodule, "graph", None)
        if graph is not None:
            yield submodule, graph


def _move_devices_fallback(exported_program, device):
    """Rewrite torch.device literals without move_to_device_pass.

    Used on torch builds that predate the pass.  Covers the same ground for
    graph kwargs; state dict and constants are handled by the caller.
    """
    for submodule, graph in _iter_graphs(exported_program):
        touched = False
        for node in graph.nodes:
            kwargs = dict(node.kwargs)
            for key, value in kwargs.items():
                if isinstance(value, torch.device) and value.type != device.type:
                    kwargs[key] = device
                    touched = True
            if touched:
                node.kwargs = kwargs
        if touched:
            submodule.recompile()
    return exported_program


def _retarget_autocast(exported_program, device_type):
    """Point wrap_with_autocast regions at ``device_type``.

    Returns the number of nodes rewritten.
    """
    rewritten = 0
    for submodule, graph in _iter_graphs(exported_program):
        touched = False
        for node in graph.nodes:
            if "autocast" not in str(node.target):
                continue
            new_args = tuple(
                device_type if (isinstance(a, str) and a in _DEVICE_STRINGS) else a
                for a in node.args
            )
            if new_args != node.args:
                node.args = new_args
                touched = True
                rewritten += 1
        if touched:
            submodule.recompile()
    return rewritten


def retarget_exported_program(exported_program, device):
    """Rewrite a CPU-traced ExportedProgram to run on ``device``.

    Safe to call when the program is already on the target device (both steps
    become no-ops).  ROCm needs no special case: HIP reports itself as ``cuda``.

    Do NOT call this on a torch_tensorrt ``.ep`` program -- those are compiled
    for a specific CUDA target and cannot be moved.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if move_to_device_pass is not None:
        exported_program = move_to_device_pass(exported_program, device)
    else:
        exported_program = _move_devices_fallback(exported_program, device)

    _retarget_autocast(exported_program, device.type)
    return exported_program


def load_exported_module(path_or_buffer, device):
    """Load a .pt2 archive and return a module ready to run on ``device``."""
    from torch.export import load as export_load

    try:
        exported_program = export_load(path_or_buffer)
    except (AssertionError, RuntimeError) as exc:
        message = str(exc)
        if "CUDA" in message or "cuda" in message:
            raise RuntimeError(
                f"Failed to load {path_or_buffer!r}: this model was exported on CUDA, so its "
                f"weights can only be materialized on a CUDA device. Update to a current model "
                f"build (exported on CPU) to run on {device}. Original error: {message}"
            ) from exc
        raise

    exported_program = retarget_exported_program(exported_program, device)
    return exported_program.module().to(device)
