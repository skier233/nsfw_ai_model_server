import logging
import time
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import _load_image_tensor
from lib.pipeline.preprocess_spec import PreprocessSpec, apply_spec


class ImagePreprocessorModel(Model):
    """Spec-driven image preprocessor.

    The ``specs`` list (set at pipeline construction time by the dynamic AI
    manager) declares every preprocessed tensor this pipeline needs.
    A single disk read produces all of them — one per spec — enabling
    models at different resolutions/formats to share the same I/O.
    """

    def __init__(self, configValues):
        super().__init__(configValues)
        self.logger = logging.getLogger("logger")

        # Populated by the dynamic_ai_manager at pipeline construction.
        # Each entry is a PreprocessSpec; output_names[i] corresponds to
        # specs[i].
        self.specs: list[PreprocessSpec] = []

    async def worker_function(self, data):
        for item in data:
            try:
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                start_time = time.perf_counter()

                # One disk read → CPU uint8 CHW tensor.
                raw_uint8 = _load_image_tensor(input_data)
                # Float conversion once — all specs scale from this.
                raw_float = raw_uint8.float()  # [0, 255] fp32 CPU
                del raw_uint8

                # Produce a tensor for each spec.
                # Specs are pre-sorted (highest effective resolution first)
                # by the dynamic_ai_manager, so output ordering is stable.
                for spec, output_key in zip(self.specs, item.output_names):
                    tensor = apply_spec(raw_float, spec)
                    await itemFuture.set_data(output_key, tensor)

                del raw_float

                elapsed = time.perf_counter() - start_time
                root_future = getattr(itemFuture, "root_future", itemFuture)
                metrics = getattr(root_future, "_pipeline_metrics", None)
                if metrics is None:
                    metrics = {}
                    setattr(root_future, "_pipeline_metrics", metrics)
                metrics["preprocess_seconds"] = metrics.get("preprocess_seconds", 0.0) + elapsed
                metrics["images_preprocessed"] = metrics.get("images_preprocessed", 0) + 1
                metrics["preprocess_backend"] = "image_preprocessor"
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