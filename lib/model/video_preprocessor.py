import asyncio
import logging
import queue
import threading
import time

import torch
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import (
    preprocess_video_deffcode,
    preprocess_video_deffcode_gpu,
    preprocess_video_deffcode_auto,
    preprocess_video_av,
    preprocess_video_av_seek,
)
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec


class VideoPreprocessorModel(Model):
    """Spec-driven video preprocessor.

    The ``specs`` list (set at pipeline construction time by the dynamic AI
    manager) declares every preprocessed tensor the pipeline needs *per
    frame*.  A single video decode pass produces all of them.

    Fixed per-frame outputs (always present):
        output_names[0] → ``dynamic_children`` (parent-level: list of child futures)
        output_names[1] → ``frame_index``
        output_names[2] → ``dynamic_threshold``
        output_names[3] → ``dynamic_return_confidence``
        output_names[4] → ``dynamic_skipped_categories``

    Spec-driven outputs (one per spec, starting at index ``FIXED_OUTPUT_COUNT``):
        output_names[FIXED_OUTPUT_COUNT + i] → tensor for ``specs[i]``
    """

    FIXED_OUTPUT_COUNT = 5  # children, frame_index, threshold, return_confidence, skipped_categories

    def __init__(self, configValues):
        super().__init__(configValues)
        self.frame_interval = configValues.get("frame_interval", 0.5)
        self.logger = logging.getLogger("logger")

        # Populated by the dynamic_ai_manager at pipeline construction.
        self.specs: list[PreprocessSpec] = []

        # Limit how many preprocessed frames can be in-flight before the
        # preprocessor pauses to let GPU inference catch up.  Prevents
        # RAM exhaustion on long videos with heavy pipelines (e.g. face).
        # 0 or null = unlimited.
        _max_pending = configValues.get("max_pending_frames", 0)
        self._max_pending_frames = int(_max_pending) if _max_pending else 0

        # When using deffcode_auto, GPU decoding is chosen only when the
        # video's longest edge is >= this threshold.  0 = always GPU.
        self._gpu_min_long_edge = int(configValues.get("gpu_min_long_edge", 3600))

        requested_backend = str(configValues.get("preprocess_backend", "deffcode_auto")).lower()

        if requested_backend == "av":
            self._preprocess_backend = "av"
            self._preprocess_callable = preprocess_video_av
            self.logger.info("Video preprocessor using PyAV threaded backend")
        elif requested_backend == "av_seek":
            self._preprocess_backend = "av_seek"
            self._preprocess_callable = preprocess_video_av_seek
            self.logger.info("Video preprocessor using PyAV seek backend")
        elif requested_backend == "deffcode_gpu":
            if not torch.cuda.is_available():
                self.logger.warning(
                    "CUDA is not available; falling back to DeFFcode CPU backend for video preprocessing"
                )
                self._preprocess_backend = "deffcode"
                self._preprocess_callable = preprocess_video_deffcode
            else:
                self._preprocess_backend = "deffcode_gpu"
                self._preprocess_callable = preprocess_video_deffcode_gpu
                self.logger.info("Video preprocessor using DeFFcode GPU backend")
        elif requested_backend == "deffcode":
            self._preprocess_backend = "deffcode"
            self._preprocess_callable = preprocess_video_deffcode
            self.logger.info("Video preprocessor using DeFFcode CPU backend")
        else:
            # Default: deffcode_auto — picks GPU or CPU per-video based on resolution.
            if not torch.cuda.is_available():
                self.logger.info(
                    "DeFFcode Auto selected and CUDA is not available; using DeFFcode CPU backend"
                )
                self._preprocess_backend = "deffcode"
                self._preprocess_callable = preprocess_video_deffcode
            else:
                self._preprocess_backend = "deffcode_auto"
                self._preprocess_callable = preprocess_video_deffcode_auto
                self.logger.info(
                    "Video preprocessor using DeFFcode Auto backend (gpu_min_long_edge=%d)",
                    self._gpu_min_long_edge,
                )

    async def worker_function(self, data):
        for item in data:
            try:
                preprocess_time = 0.0
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                use_timestamps = itemFuture[item.input_names[1]]
                frame_interval = itemFuture[item.input_names[2]] or self.frame_interval
                vr_video = itemFuture[item.input_names[5]]
                children = []
                frame_count = 0
                preprocess_callable = self._preprocess_callable
                backend_used = self._preprocess_backend
                # Determine the max decode resolution from the specs.
                # If every spec has a finite cap we can let ffmpeg downscale
                # at decode time → dramatically less per-frame data.
                _spec_edges = [s.effective_resolution for s in self.specs]
                _max_decode_long_edge = 0
                if _spec_edges and all(e < 999_999 for e in _spec_edges):
                    _max_decode_long_edge = max(_spec_edges)

                # Decode at native (or capped) resolution — apply_spec handles
                # per-model resize/normalize/device.  With norm_config=-1 the
                # backends yield (frame_index, tensor) where tensor is a
                # [0,255] float32 CHW tensor — exactly what apply_spec expects.
                try:
                    frame_source = preprocess_callable(
                        input_data,
                        frame_interval,
                        0,             # image_size=0: no resize at decode level
                        False,         # use_half_precision=False: fp32 base
                        "cpu",         # device: keep on CPU — apply_spec handles resize/norm
                        use_timestamps,
                        vr_video=vr_video,
                        norm_config=-1,            # skip normalization
                        max_decode_long_edge=_max_decode_long_edge,
                        gpu_min_long_edge=self._gpu_min_long_edge,
                    )
                except Exception as exc:
                    if preprocess_callable is preprocess_video_deffcode_gpu:
                        self.logger.warning(
                            "DeFFcode GPU preprocessing failed for '%s'. Falling back to DeFFcode CPU. Error: %s",
                            input_data, exc,
                        )
                        preprocess_callable = preprocess_video_deffcode
                        backend_used = "deffcode"
                        frame_source = preprocess_callable(
                            input_data, frame_interval, 0, False, "cpu",
                            use_timestamps, vr_video=vr_video, norm_config=-1,
                            max_decode_long_edge=_max_decode_long_edge,
                        )
                    elif preprocess_callable is preprocess_video_deffcode_auto:
                        self.logger.warning(
                            "DeFFcode Auto preprocessing failed for '%s'. Falling back to DeFFcode CPU. Error: %s",
                            input_data, exc,
                        )
                        preprocess_callable = preprocess_video_deffcode
                        backend_used = "deffcode"
                        frame_source = preprocess_callable(
                            input_data, frame_interval, 0, False, "cpu",
                            use_timestamps, vr_video=vr_video, norm_config=-1,
                            max_decode_long_edge=_max_decode_long_edge,
                        )
                    else:
                        raise

                spec_start = self.FIXED_OUTPUT_COUNT
                frame_semaphore = None
                if self._max_pending_frames > 0:
                    frame_semaphore = asyncio.Semaphore(self._max_pending_frames)
                frame_iterator = iter(frame_source)
                loop = asyncio.get_running_loop()

                # ---- Continuous producer-consumer pipeline ----
                #
                # Previously, decode→transform→apply_spec was done one
                # frame at a time via ``await run_in_executor()``, which
                # forced an event-loop round-trip between every frame
                # (~10 ms overhead × 1863 frames = ~19 s wasted).
                #
                # Now a background thread runs the full loop without
                # ever yielding to the event loop.  Results go into a
                # bounded queue; the async consumer just pulls them out
                # and creates ItemFutures.  Three-stage pipeline:
                #
                #   Thread A (av_seek prefetch): seek+decode (GIL-free)
                #   Thread B (producer below) : transform+apply_spec
                #   Async consumer            : ItemFuture → GPU
                #
                # All three stages overlap.  Total ≈ max(decode, GPU).

                _FRAME_DONE = object()
                _QUEUE_DEPTH = 64
                _result_q = queue.Queue(maxsize=_QUEUE_DEPTH)
                _producer_error = []
                _cumulative_cpu = [0.0]   # mutable float for thread
                _timing = {
                    "decode_wait": 0.0,   # time blocked on next(frame_iterator)
                    "apply_spec": 0.0,    # time in apply_spec
                    "queue_put": 0.0,     # time blocked on queue.put
                    "frames": 0,
                }

                def _frame_producer():
                    # Build CPU-only variants of specs so that apply_spec
                    # never touches the GPU.  Individual H2D transfers per
                    # frame on the default CUDA stream serialize with GPU
                    # inference and cause the spiky utilisation pattern.
                    # Keeping everything on CPU lets the batched inference
                    # step do a single large H2D copy instead.
                    from dataclasses import replace as _dc_replace
                    _specs = [
                        _dc_replace(sp, device="cpu") if sp.device != "cpu" else sp
                        for sp in self.specs
                    ]
                    _out_names = item.output_names
                    _ss = spec_start
                    _t = _timing
                    try:
                        while True:
                            t_dec_start = time.perf_counter()
                            try:
                                frame_data = next(frame_iterator)
                            except StopIteration:
                                break
                            t_dec_end = time.perf_counter()
                            _t["decode_wait"] += t_dec_end - t_dec_start

                            frame_index = frame_data[0]
                            raw_frame = frame_data[1]
                            del frame_data

                            t0 = time.perf_counter()
                            st = {}
                            for i, sp in enumerate(_specs):
                                st[_out_names[_ss + i]] = apply_spec(raw_frame, sp)
                            del raw_frame
                            t1 = time.perf_counter()

                            _result_q.put((frame_index, st))
                            t2 = time.perf_counter()

                            _t["apply_spec"] += t1 - t0
                            _t["queue_put"] += t2 - t1
                            _t["frames"] += 1

                            # Mark the start of the next decode wait
                            _cumulative_cpu[0] += t1 - t0
                    except Exception as exc:
                        _producer_error.append(exc)
                    finally:
                        _result_q.put(_FRAME_DONE)

                producer = threading.Thread(
                    target=_frame_producer, daemon=True,
                    name="video-preprocess-producer",
                )
                producer.start()

                try:
                    # Pull frames from the producer in batches to reduce
                    # thread-pool round-trips (1 call per batch of frames
                    # instead of 1 call per frame).
                    _threshold = itemFuture[item.input_names[3]]
                    _return_conf = itemFuture[item.input_names[4]]
                    _skipped_cats = itemFuture[item.input_names[6]]
                    _out_names = item.output_names
                    _consumer_timing = {
                        "pull_wait": 0.0,     # time blocked waiting for producer
                        "create_futures": 0.0, # time creating ItemFutures
                        "pulls": 0,
                        "pull_sizes": [],
                    }

                    def _pull_batch():
                        """Block for the first item, then drain non-blocking."""
                        first = _result_q.get()
                        if first is _FRAME_DONE:
                            return [first]
                        items = [first]
                        # Drain up to 63 more without blocking
                        while len(items) < 64:
                            try:
                                nxt = _result_q.get_nowait()
                                items.append(nxt)
                                if nxt is _FRAME_DONE:
                                    break
                            except queue.Empty:
                                break
                        return items

                    while True:
                        t_pull_start = time.perf_counter()
                        batch = await loop.run_in_executor(None, _pull_batch)
                        t_pull_end = time.perf_counter()
                        _consumer_timing["pull_wait"] += t_pull_end - t_pull_start
                        _consumer_timing["pulls"] += 1
                        _consumer_timing["pull_sizes"].append(len(batch))

                        hit_done = False
                        t_create_start = time.perf_counter()
                        for result in batch:
                            if result is _FRAME_DONE:
                                hit_done = True
                                break

                            frame_index, spec_tensors = result
                            frame_count += 1

                            payload = {
                                _out_names[1]: frame_index,
                                _out_names[2]: _threshold,
                                _out_names[3]: _return_conf,
                                _out_names[4]: _skipped_cats,
                            }
                            payload.update(spec_tensors)
                            if frame_semaphore is not None:
                                await frame_semaphore.acquire()
                            child = await ItemFuture.create(item, payload, item.item_future.handler)
                            if frame_semaphore is not None:
                                child.future.add_done_callback(lambda _, s=frame_semaphore: s.release())
                            children.append(child)
                        _consumer_timing["create_futures"] += time.perf_counter() - t_create_start

                        if hit_done:
                            if _producer_error:
                                raise _producer_error[0]
                            break

                    preprocess_time = _cumulative_cpu[0]
                    # --- Timing summary ---
                    producer.join(timeout=10.0)
                    _t = _timing
                    _ct = _consumer_timing
                    avg_pull = sum(_ct["pull_sizes"]) / max(len(_ct["pull_sizes"]), 1)
                    self.logger.warning(
                        "PIPELINE TIMING [%d frames] — "
                        "Producer: decode_wait=%.2fs, apply_spec=%.2fs, queue_put=%.2fs | "
                        "Consumer: pull_wait=%.2fs, create_futures=%.2fs, pulls=%d, avg_pull_size=%.1f | "
                        "Total producer=%.2fs",
                        _t["frames"],
                        _t["decode_wait"], _t["apply_spec"], _t["queue_put"],
                        _ct["pull_wait"], _ct["create_futures"], _ct["pulls"], avg_pull,
                        _t["decode_wait"] + _t["apply_spec"] + _t["queue_put"],
                    )
                finally:
                    # Drain queue so the producer isn't stuck on put().
                    while producer.is_alive():
                        try:
                            _result_q.get_nowait()
                        except queue.Empty:
                            break
                    producer.join(timeout=5.0)
                    close = getattr(frame_source, "close", None)
                    if callable(close):
                        close()

                if frame_count > 0:
                    avg_time = preprocess_time / frame_count
                    root_future = getattr(itemFuture, "root_future", itemFuture)
                    metrics = getattr(root_future, "_pipeline_metrics", None)
                    if metrics is None:
                        metrics = {}
                        setattr(root_future, "_pipeline_metrics", metrics)
                    metrics["preprocess_seconds"] = metrics.get("preprocess_seconds", 0.0) + preprocess_time
                    metrics["frames_preprocessed"] = metrics.get("frames_preprocessed", 0) + frame_count
                    metrics["preprocess_backend"] = backend_used
                    metrics["average_frame_preprocess_seconds"] = avg_time
                    self.logger.info(
                        "Preprocessed %s frames in %.4f seconds (avg %.4f s/frame) using %s backend.",
                        frame_count, preprocess_time, avg_time, backend_used,
                    )
                else:
                    error_msg = f"No frames were produced during preprocessing of '{input_data}' using {backend_used} backend."
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

                await itemFuture.set_data(item.output_names[0], children)
            except FileNotFoundError as fnf_error:
                self.logger.error(f"File not found error: {fnf_error}")
                self.logger.debug("Stack trace:", exc_info=True)
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                self.logger.error(f"IO error (video might be corrupted): {io_error}")
                self.logger.debug("Stack trace:", exc_info=True)
                itemFuture.set_exception(io_error)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                itemFuture.set_exception(e)