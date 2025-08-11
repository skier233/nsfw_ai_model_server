import logging
import time
import torch

from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import preprocess_video


class VideoPreprocessorModel(Model):
    def __init__(self, configValues):
        super().__init__(configValues)
        self.image_size = configValues.get("image_size", 512)
        self.frame_interval = configValues.get("frame_interval", 0.5)
        self.use_half_precision = configValues.get("use_half_precision", True)
        self.device = configValues.get("device", None)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.logger = logging.getLogger("logger")

        if isinstance(self.device, str) and self.device:
            self.device = torch.device(self.device)

    async def worker_function(self, data):
        for item in data:
            try:
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                use_timestamps = itemFuture[item.input_names[1]]
                frame_interval = itemFuture[item.input_names[2]] or self.frame_interval
                vr_video = itemFuture[item.input_names[5]]
                norm_config = self.normalization_config or 1

                children = []
                frames_processed = 0
                total_time = 0.0
                t_prev = time.time()

                for frame_index, frame in preprocess_video(
                    input_data,
                    frame_interval,
                    self.image_size,
                    self.use_half_precision,
                    self.device,
                    use_timestamps,
                    vr_video=vr_video,
                    norm_config=norm_config
                ):
                    # Ensure on CUDA if requested (no-op if already there)
                    if isinstance(self.device, torch.device) and self.device.type == "cuda" and hasattr(frame, "to"):
                        try:
                            if frame.device.type == "cpu":
                                frame = frame.pin_memory()
                        except Exception:
                            pass
                        frame = frame.to(self.device, non_blocking=True)
                        if self.use_half_precision and frame.dtype == torch.float32:
                            frame = frame.half()

                    now = time.time()
                    total_time += (now - t_prev)
                    t_prev = now
                    frames_processed += 1

                    payload = {
                        item.output_names[1]: frame,
                        item.output_names[2]: frame_index,
                        item.output_names[3]: itemFuture[item.input_names[3]],
                        item.output_names[4]: itemFuture[item.input_names[4]],
                        item.output_names[5]: itemFuture[item.input_names[6]],
                    }
                    result = await ItemFuture.create(item, payload, item.item_future.handler)
                    children.append(result)

                if frames_processed > 0:
                    avg = total_time / frames_processed
                    self.logger.info(
                        f"Preprocessed {frames_processed} frames in {total_time:.3f}s "
                        f"({avg:.4f}s/frame, {1.0 / avg:.2f} FPS)."
                    )
                else:
                    self.logger.info("Preprocessed 0 frames.")

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
