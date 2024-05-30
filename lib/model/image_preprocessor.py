import logging
import time
from ai_processing import preprocess_image
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model


class ImagePreprocessorModel(Model):
    def __init__(self, configValues):
        super().__init__(configValues)  # Assuming Python 3 syntax for brevity
        self.image_size = configValues.get("image_size", 512)
        self.use_half_precision = configValues.get("use_half_precision", True)
        self.device = configValues.get("device", None)
        self.logger = logging.getLogger("logger")
    
    async def worker_function(self, data):
        for item in data:
            try:
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                preprocessed_frame = preprocess_image(input_data, self.image_size, self.use_half_precision, self.device)
                await itemFuture.set_data(item.output_names[0], preprocessed_frame)
            except FileNotFoundError as fnf_error:
                self.logger.error(f"File not found error: {fnf_error}")
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                self.logger.error(f"IO error (image might be corrupted): {io_error}")
                itemFuture.set_exception(io_error)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                itemFuture.set_exception(e)