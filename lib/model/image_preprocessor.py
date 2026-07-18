import asyncio
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

    Disk I/O and tensor transforms run in a thread pool so the event loop
    stays free for AI model batch collection.
    """

    def __init__(self, configValues):
        super().__init__(configValues)
        self.logger = logging.getLogger("logger")

        # Populated by the dynamic_ai_manager at pipeline construction.
        # Each entry is a PreprocessSpec; output_names[i] corresponds to
        # specs[i].
        self.specs: list[PreprocessSpec] = []

        # Limit how many preprocessed images can be in-flight before the
        # preprocessor pauses to let GPU inference catch up.  Prevents
        # RAM/VRAM exhaustion on large batch requests.
        # 0 or null = unlimited.
        _max_pending = configValues.get("max_pending_images", 0)
        self._max_pending_images = int(_max_pending) if _max_pending else 0

    async def worker_function(self, data):
        loop = asyncio.get_running_loop()
        semaphore = None
        if self._max_pending_images > 0:
            semaphore = asyncio.Semaphore(self._max_pending_images)

        for item in data:
            try:
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]

                if semaphore is not None:
                    await semaphore.acquire()

                start_time = time.perf_counter()

                # Run disk I/O + resize/normalize in thread pool so the
                # event loop stays free for AI model batch collection.
                spec_tensors = await loop.run_in_executor(
                    None, _load_and_apply_specs,
                    input_data, self.specs, item.output_names,
                )

                for output_key, tensor in spec_tensors.items():
                    await itemFuture.set_data(output_key, tensor)

                if semaphore is not None:
                    semaphore.release()

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


def _load_and_apply_specs(image_path, specs, output_names):
    """Load an image from disk and apply all preprocess specs.

    Runs in a thread pool — must not touch asyncio primitives.
    Returns a dict of {output_key: tensor}.
    """
    raw_uint8 = _load_image_tensor(image_path)
    raw_float = raw_uint8.float()  # [0, 255] fp32 CPU
    del raw_uint8

    spec_tensors = {}
    for spec, output_key in zip(specs, output_names):
        spec_tensors[output_key] = apply_spec(raw_float, spec)

    del raw_float
    return spec_tensors