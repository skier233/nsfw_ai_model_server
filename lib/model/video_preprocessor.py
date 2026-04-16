import asyncio
import logging
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
                        None,          # device
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
                            input_data, frame_interval, 0, False, None,
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
                            input_data, frame_interval, 0, False, None,
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

                _SENTINEL = object()  # signals iterator exhaustion from thread

                def _decode_and_apply_specs(frame_iterator, specs, output_names, spec_start):
                    """Decode one frame and apply all specs (CPU-bound work).

                    Runs in a thread pool so the event loop stays free for
                    AI model batch collection during video processing.
                    Returns _SENTINEL when the iterator is exhausted.
                    """
                    frame_data = next(frame_iterator, _SENTINEL)
                    if frame_data is _SENTINEL:
                        return _SENTINEL

                    frame_index = frame_data[0]
                    raw_frame = frame_data[1]
                    del frame_data

                    if raw_frame.dtype != torch.half:
                        raw_frame = raw_frame.half()

                    spec_tensors = {}
                    for i, spec in enumerate(specs):
                        spec_tensors[output_names[spec_start + i]] = apply_spec(raw_frame, spec)

                    del raw_frame
                    return frame_index, spec_tensors

                try:
                    while True:
                        chunk_start = time.perf_counter()
                        result = await loop.run_in_executor(
                            None, _decode_and_apply_specs,
                            frame_iterator, self.specs, item.output_names, spec_start,
                        )
                        if result is _SENTINEL:
                            break
                        frame_index, spec_tensors = result
                        preprocess_time += time.perf_counter() - chunk_start

                        frame_count += 1

                        # Build per-frame child payload.
                        payload = {
                            item.output_names[1]: frame_index,
                            item.output_names[2]: itemFuture[item.input_names[3]],
                            item.output_names[3]: itemFuture[item.input_names[4]],
                            item.output_names[4]: itemFuture[item.input_names[6]],
                        }
                        payload.update(spec_tensors)
                        if frame_semaphore is not None:
                            await frame_semaphore.acquire()
                        result = await ItemFuture.create(item, payload, item.item_future.handler)
                        if frame_semaphore is not None:
                            result.future.add_done_callback(lambda _, s=frame_semaphore: s.release())
                        children.append(result)
                finally:
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