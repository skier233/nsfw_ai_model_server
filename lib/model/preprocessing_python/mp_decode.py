"""Multiprocess PyAV video decode.

This module is deliberately lightweight — it imports only ``av`` and ``numpy``
(no torch / no server packages).  On Windows ``spawn`` the child processes
import *this* module to locate the worker function, so keeping it torch-free
keeps process startup fast (~tens of ms instead of seconds).

The strategy: split the timeline into contiguous chunks and decode each chunk
in its own process.  Because each process has its own GIL, the decoders run in
true parallel AND don't contend with the parent's asyncio inference pipeline —
the thing that throttles in-process (threaded) PyAV decode.  Only the small,
already-downscaled frames cross the process boundary.
"""

import math
import multiprocessing as mp
from typing import Optional


def _decode_chunk_worker(path, start_idx, count, frame_interval, max_long_edge,
                         use_timestamps, out_queue):
    """Decode one contiguous chunk and push (out_idx, HWC uint8 ndarray) tuples.

    Runs in its own process.  Emits the frame nearest each target timestamp
    (first frame whose time is >= target), giving true per-interval sampling.
    """
    try:
        import av

        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            time_base = stream.time_base
            average_rate = float(stream.average_rate) if stream.average_rate else 30.0
            tol = 0.5 / average_rate if average_rate > 0 else 0.02

            start_t = start_idx * frame_interval
            container.seek(int(start_t / time_base), stream=stream)

            produced = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                ft = float(frame.pts * time_base)
                target = (start_idx + produced) * frame_interval
                if ft + tol < target:
                    continue  # decoded (unavoidable) but before the next target
                if max_long_edge > 0 and max(frame.width, frame.height) > max_long_edge:
                    scale = max_long_edge / max(frame.width, frame.height)
                    tw = max(2, round(frame.width * scale))
                    th = max(2, round(frame.height * scale))
                    arr = frame.reformat(width=tw, height=th, format="rgb24").to_ndarray()
                else:
                    arr = frame.to_ndarray(format="rgb24")
                out_idx = target if use_timestamps else (start_idx + produced)
                out_queue.put((out_idx, arr))
                produced += 1
                if produced >= count:
                    break
        finally:
            container.close()
    except Exception as exc:  # surface to the parent as a sentinel payload
        try:
            out_queue.put(("__error__", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        out_queue.put(None)  # done sentinel


def probe_duration_seconds(path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return container.duration / av.time_base
    return 0.0


def iter_parallel_frames(path, frame_interval, max_long_edge, decode_workers,
                         use_timestamps, queue_depth: int = 64):
    """Yield (out_idx, HWC uint8 ndarray) from parallel decode worker processes.

    Frames arrive out of timestamp order; each carries its target time/index.
    """
    duration = probe_duration_seconds(path)
    if duration <= 0 or not frame_interval or frame_interval <= 0:
        return

    n_targets = int(duration / frame_interval) + 1
    n_workers = max(1, min(int(decode_workers), n_targets))
    per = math.ceil(n_targets / n_workers)

    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue(maxsize=queue_depth)

    procs = []
    for i in range(n_workers):
        start_idx = i * per
        count = min(per, n_targets - start_idx)
        if count <= 0:
            continue
        proc = ctx.Process(
            target=_decode_chunk_worker,
            args=(str(path), start_idx, count, frame_interval, int(max_long_edge or 0),
                  bool(use_timestamps), out_queue),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    finished = 0
    error = None
    try:
        while finished < len(procs):
            item = out_queue.get()
            if item is None:
                finished += 1
                continue
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                error = RuntimeError(f"parallel decode worker failed: {item[1]}")
                break
            yield item
        if error is not None:
            raise error
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=2.0)
