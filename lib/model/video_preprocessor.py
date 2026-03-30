import logging
import time

import torch
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import (
    preprocess_video,
    preprocess_video_deffcode,
    preprocess_video_deffcode_gpu,
    preprocess_video_deffcode_auto,
    get_normalization_config,
    get_frame_transforms,
)

class VideoPreprocessorModel(Model):
    def __init__(self, configValues):
        super().__init__(configValues)
        self.image_size = configValues.get("image_size", 512)
        self.frame_interval = configValues.get("frame_interval", 0.5)
        self.use_half_precision = configValues.get("use_half_precision", True)
        self.device = configValues.get("device", None)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.logger = logging.getLogger("logger")

        # When True the preprocessor skips the base frame transform
        # (resize + normalize for tagging models) and only produces the
        # frames described by extra_frame_specs.  The base output slot
        # (output_names[1]) is set to None so no memory is wasted.
        # The dynamic_ai_manager sets this when the pipeline has no
        # full-image (tagging) models.
        self.skip_base_frame = False

        # Extra per-model frame specs set by the dynamic AI manager.
        # Each entry: {"output_index": int, "image_size": int,
        #              "norm_config": int, "use_half": bool}
        # When present the preprocessor captures the base CHW tensor from
        # each decoded frame and applies each spec's transforms to produce
        # additional model-resolution-matched outputs.
        self.extra_frame_specs = []

        requested_backend = str(configValues.get("preprocess_backend", "deffcode_auto")).lower()

        self._preprocess_backend = "decord"
        self._preprocess_callable = preprocess_video

        if requested_backend in {"deffcode_gpu", "deffcode", "deffcode_auto"}:
            backend_choice = requested_backend
            if backend_choice == "deffcode_gpu" and not torch.cuda.is_available():
                self.logger.warning(
                    "CUDA is not available; falling back to DeFFcode CPU backend for video preprocessing"
                )
                backend_choice = "deffcode"
            elif backend_choice == "deffcode_auto" and not torch.cuda.is_available():
                self.logger.info(
                    "DeFFcode Auto selected and CUDA is not available; falling back to DeFFcode CPU backend for video preprocessing"
                )
                backend_choice = "deffcode"

            if backend_choice == "deffcode_gpu":
                self._preprocess_backend = "deffcode_gpu"
                self._preprocess_callable = preprocess_video_deffcode_gpu
                self.logger.info("Video preprocessor using DeFFcode GPU backend")
            elif backend_choice == "deffcode_auto":
                self._preprocess_backend = "deffcode_auto"
                self._preprocess_callable = preprocess_video_deffcode_auto
                self.logger.info("Video preprocessor using DeFFcode Auto backend")
            else:
                self._preprocess_backend = "deffcode"
                self._preprocess_callable = preprocess_video_deffcode
                self.logger.info("Video preprocessor using DeFFcode backend")
        else:
            self.logger.info("Video preprocessor using Decord backend")
    
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
                norm_config = self.normalization_config or 1
                frame_count = 0
                preprocess_callable = self._preprocess_callable
                backend_used = self._preprocess_backend

                # When extra specs are configured, request the base CHW
                # tensor from the preprocess function so we can apply
                # model-resolution-matched transforms per spec.
                _extra_specs = self.extra_frame_specs or []
                include_base = bool(_extra_specs)

                # Pre-build extra transform pipelines (cheap — runs once)
                _extra_transforms = []
                if include_base:
                    _resolve_device = torch.device(self.device) if self.device else torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    for spec in _extra_specs:
                        s_mean, s_std = get_normalization_config(spec["norm_config"], _resolve_device)
                        s_raw = (spec["norm_config"] == -1)
                        s_xform = get_frame_transforms(
                            spec["use_half"],
                            s_mean,
                            s_std,
                            img_size=spec["image_size"],
                            apply_resize=(spec["image_size"] > 0),
                            scale_values=(not s_raw),
                        )
                        _extra_transforms.append((
                            spec["output_index"],
                            s_xform,
                            spec.get("max_long_edge", 0),
                            spec.get("force_cpu", False),
                            _resolve_device if not spec.get("force_cpu", False) else None,
                        ))

                # When skip_base_frame is set we don't need the
                # classification tensor at all — only the raw CHW base.
                # Use CPU device and raw mode so we skip the expensive
                # GPU transfer + resize + normalise per frame.
                _preproc_device = self.device
                _preproc_img_size = self.image_size
                _preproc_norm = norm_config
                if self.skip_base_frame and include_base:
                    _preproc_device = 'cpu'
                    _preproc_img_size = 0
                    _preproc_norm = -1

                # Compute the maximum resolution anything downstream
                # actually needs.  If all consumers have a max_long_edge
                # cap we can tell FFmpeg to downscale during decode —
                # dramatically reducing per-frame data size.
                _max_decode_long_edge = 0
                if include_base and self.skip_base_frame:
                    # No classification frame needed → decode only needs
                    # to cover the largest extra spec cap.
                    _spec_edges = [s.get("max_long_edge", 0) for s in _extra_specs]
                    if _spec_edges and all(e > 0 for e in _spec_edges):
                        _max_decode_long_edge = max(_spec_edges)

                try:
                    frame_source = preprocess_callable(
                        input_data,
                        frame_interval,
                        _preproc_img_size,
                        self.use_half_precision,
                        _preproc_device,
                        use_timestamps,
                        vr_video=vr_video,
                        norm_config=_preproc_norm,
                        include_raw=include_base,
                        max_decode_long_edge=_max_decode_long_edge,
                    )
                except Exception as exc:
                    if preprocess_callable is preprocess_video_deffcode_gpu:
                        self.logger.warning(
                            "DeFFcode GPU preprocessing failed for '%s'. Falling back to DeFFcode CPU. Error: %s",
                            input_data,
                            exc,
                        )
                        preprocess_callable = preprocess_video_deffcode
                        backend_used = "deffcode"
                        frame_source = preprocess_callable(
                            input_data,
                            frame_interval,
                            _preproc_img_size,
                            self.use_half_precision,
                            _preproc_device,
                            use_timestamps,
                            vr_video=vr_video,
                            norm_config=_preproc_norm,
                            include_raw=include_base,
                            max_decode_long_edge=_max_decode_long_edge,
                        )
                    elif preprocess_callable is preprocess_video_deffcode:
                        self.logger.warning(
                            "DeFFcode preprocessing failed for '%s'. Falling back to Decord. Error: %s",
                            input_data,
                            exc,
                        )
                        preprocess_callable = preprocess_video
                        backend_used = "decord"
                        frame_source = preprocess_callable(
                            input_data,
                            frame_interval,
                            _preproc_img_size,
                            self.use_half_precision,
                            _preproc_device,
                            use_timestamps,
                            vr_video=vr_video,
                            norm_config=_preproc_norm,
                            include_raw=include_base,
                            max_decode_long_edge=_max_decode_long_edge,
                        )
                    else:
                        raise

                frame_iterator = iter(frame_source)
                try:
                    while True:
                        chunk_start = time.perf_counter()
                        try:
                            frame_data = next(frame_iterator)
                        except StopIteration:
                            break
                        preprocess_time += time.perf_counter() - chunk_start
                        frame_count += 1

                        frame_index = frame_data[0]
                        frame = frame_data[1]

                        payload = {
                            item.output_names[2]: frame_index,
                            item.output_names[3]: itemFuture[item.input_names[3]],
                            item.output_names[4]: itemFuture[item.input_names[4]],
                            item.output_names[5]: itemFuture[item.input_names[6]],
                        }

                        # Only store the base classification frame when
                        # tagging models need it.  When skip_base_frame is
                        # set (no full-image models in the pipeline) we
                        # release it immediately to save memory.
                        if not self.skip_base_frame:
                            payload[item.output_names[1]] = frame
                        del frame

                        # Apply model-resolution-matched transforms for each
                        # extra spec (e.g. face detector @ 640×640 [0-255]).
                        if include_base and len(frame_data) > 2:
                            base_tensor = frame_data[2]
                            for out_idx, xform, max_long_edge, force_cpu, target_device in _extra_transforms:
                                t = xform(base_tensor.clone())
                                # Aspect-ratio-preserving resize to cap memory.
                                if max_long_edge and max_long_edge > 0:
                                    h, w = t.shape[-2], t.shape[-1]
                                    long_edge = max(h, w)
                                    if long_edge > max_long_edge:
                                        scale = max_long_edge / long_edge
                                        new_h = max(1, int(round(h * scale)))
                                        new_w = max(1, int(round(w * scale)))
                                        t = torch.nn.functional.interpolate(
                                            t.unsqueeze(0),
                                            size=(new_h, new_w),
                                            mode="bilinear",
                                            align_corners=False,
                                        ).squeeze(0)
                                if force_cpu:
                                    t = t.cpu()
                                elif target_device is not None and t.device != target_device:
                                    t = t.to(target_device)
                                payload[item.output_names[out_idx]] = t
                            del base_tensor  # release ref to raw frame immediately
                        result = await ItemFuture.create(item, payload, item.item_future.handler)
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
                        frame_count,
                        preprocess_time,
                        avg_time,
                        backend_used,
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