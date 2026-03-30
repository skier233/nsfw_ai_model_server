import logging
import time
import torch
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import preprocess_image

class ImagePreprocessorModel(Model):
    def __init__(self, configValues):
        super().__init__(configValues)
        self.image_size = configValues.get("image_size", 512)
        self.use_half_precision = configValues.get("use_half_precision", True)
        self.device = configValues.get("device", None)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.logger = logging.getLogger("logger")

        # When True the preprocessor skips the classification tensor
        # (output_names[0]) and only emits the raw region-source tensor.
        # The dynamic_ai_manager sets this when the pipeline has no
        # full-image (tagging) models.
        self.skip_base_output = False

        # Optional cap on the raw region-source long-edge (pixels).
        # 0 means no cap (full resolution).  Set by the dynamic_ai_manager.
        self.region_source_max_long_edge = 0
    
    async def worker_function(self, data):
        for item in data:
            try:
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                norm_config = self.normalization_config or 1
                dual_output = len(item.output_names) > 1
                skip_base = self.skip_base_output
                max_edge = self.region_source_max_long_edge
                start_time = time.perf_counter()
                if dual_output:
                    preprocessed_frame, raw_frame = preprocess_image(
                        input_data, self.image_size, self.use_half_precision,
                        self.device, norm_config=norm_config, include_raw=True,
                    )
                    # Cap region-source resolution to save memory.
                    if max_edge and max_edge > 0:
                        h, w = raw_frame.shape[-2], raw_frame.shape[-1]
                        long_edge = max(h, w)
                        if long_edge > max_edge:
                            scale = max_edge / long_edge
                            new_h = max(1, int(round(h * scale)))
                            new_w = max(1, int(round(w * scale)))
                            raw_frame = torch.nn.functional.interpolate(
                                raw_frame.unsqueeze(0),
                                size=(new_h, new_w),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                else:
                    preprocessed_frame = preprocess_image(input_data, self.image_size, self.use_half_precision, self.device, norm_config=norm_config)
                elapsed = time.perf_counter() - start_time
                root_future = getattr(itemFuture, "root_future", itemFuture)
                metrics = getattr(root_future, "_pipeline_metrics", None)
                if metrics is None:
                    metrics = {}
                    setattr(root_future, "_pipeline_metrics", metrics)
                metrics["preprocess_seconds"] = metrics.get("preprocess_seconds", 0.0) + elapsed
                metrics["images_preprocessed"] = metrics.get("images_preprocessed", 0) + 1
                metrics["preprocess_backend"] = "image_preprocessor"
                # Set the raw output first so the longer detector → region
                # chain can start while the classification model runs in
                # parallel on the processed tensor.
                if dual_output:
                    await itemFuture.set_data(item.output_names[1], raw_frame)
                if not skip_base:
                    await itemFuture.set_data(item.output_names[0], preprocessed_frame)
                del preprocessed_frame
            except FileNotFoundError as fnf_error:
                self.logger.error(f"File not found error: {fnf_error} for file: {input_data}")
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                self.logger.error(f"IO error (image might be corrupted): {io_error} for file: {input_data}")
                self.logger.debug("Stack trace:", exc_info=True)
                itemFuture.set_exception(io_error)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e} for file: {input_data}")
                self.logger.debug("Stack trace:", exc_info=True)
                itemFuture.set_exception(e)