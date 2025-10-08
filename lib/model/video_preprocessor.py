import logging
import time

import torch
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import (
    preprocess_video,
    preprocess_video_deffcode,
    preprocess_video_deffcode_gpu,
    preprocess_video_deffcode_auto
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
                totalTime = 0
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                use_timestamps = itemFuture[item.input_names[1]]
                frame_interval = itemFuture[item.input_names[2]] or self.frame_interval
                vr_video = itemFuture[item.input_names[5]]
                children = []
                oldTime = time.time()
                norm_config = self.normalization_config or 1
                frame_count = 0
                preprocess_callable = self._preprocess_callable
                backend_used = self._preprocess_backend

                try:
                    frame_source = preprocess_callable(
                        input_data,
                        frame_interval,
                        self.image_size,
                        self.use_half_precision,
                        self.device,
                        use_timestamps,
                        vr_video=vr_video,
                        norm_config=norm_config,
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
                            self.image_size,
                            self.use_half_precision,
                            self.device,
                            use_timestamps,
                            vr_video=vr_video,
                            norm_config=norm_config,
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
                            self.image_size,
                            self.use_half_precision,
                            self.device,
                            use_timestamps,
                            vr_video=vr_video,
                            norm_config=norm_config,
                        )
                    else:
                        raise

                frame_iterator = iter(frame_source)
                try:
                    for frame_index, frame in frame_iterator:
                        frame_count += 1
                        newTime = time.time()
                        totalTime += newTime - oldTime
                        oldTime = newTime
                        payload = {
                            item.output_names[1]: frame,
                            item.output_names[2]: frame_index,
                            item.output_names[3]: itemFuture[item.input_names[3]],
                            item.output_names[4]: itemFuture[item.input_names[4]],
                            item.output_names[5]: itemFuture[item.input_names[6]],
                        }
                        result = await ItemFuture.create(item, payload, item.item_future.handler)
                        children.append(result)
                finally:
                    close = getattr(frame_source, "close", None)
                    if callable(close):
                        close()

                if frame_count > 0:
                    avg_time = totalTime / frame_count
                    self.logger.info(
                        "Preprocessed %s frames in %.4f seconds (avg %.4f s/frame) using %s backend.",
                        frame_count,
                        totalTime,
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