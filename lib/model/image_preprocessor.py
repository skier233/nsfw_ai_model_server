import logging
import time
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
    
    async def worker_function(self, data):
        for item in data:
            try:
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                norm_config = self.normalization_config or 1
                preprocessed_frame = preprocess_image(input_data, self.image_size, self.use_half_precision, self.device, norm_config=norm_config)
                await itemFuture.set_data(item.output_names[0], preprocessed_frame)
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