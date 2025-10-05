import asyncio
import logging
import time
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import (
    preprocess_video,
    preprocess_video_deffcode,
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

        backend = str(configValues.get("preprocess_backend", "deffcode")).lower()
        if backend != "deffcode":
            backend = "decord"
            self.logger.info("Video preprocessor using Decord backend")
        else:
            self.logger.info("Video preprocessor using DeFFcode backend")

        self._preprocess_backend = backend
        self._preprocess_callable = preprocess_video_deffcode if backend == "deffcode" else preprocess_video
    
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
                i = -1
                oldTime = time.time()
                norm_config = self.normalization_config or 1
                for frame_index, frame in self._preprocess_callable(input_data, frame_interval, self.image_size, self.use_half_precision, self.device, use_timestamps, vr_video=vr_video, norm_config=norm_config):
                    i += 1
                    newTime = time.time()
                    totalTime += newTime - oldTime
                    oldTime = newTime
                    data = {item.output_names[1]: frame, item.output_names[2]: frame_index, item.output_names[3]: itemFuture[item.input_names[3]], item.output_names[4]: itemFuture[item.input_names[4]], item.output_names[5]: itemFuture[item.input_names[6]]}
                    result = await ItemFuture.create(item, data, item.item_future.handler)
                    children.append(result)
                self.logger.info(f"Preprocessed {i} frames in {totalTime} seconds at an average of {totalTime/i} seconds per frame.")
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