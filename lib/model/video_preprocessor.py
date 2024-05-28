import time
from ai_processing import preprocess_video
from lib.async_lib.async_processing import ItemFuture
from lib.model.model import Model


class VideoPreprocessorModel(Model):
    def __init__(self, configValues):
        super().__init__(configValues)
        self.image_size = configValues.get("image_size", 512)
        self.frame_interval = configValues.get("frame_interval", 0.5)
        self.use_half_precision = configValues.get("use_half_precision", True)
        self.device = configValues.get("device", None)
    
    async def worker_function(self, data):
        for item in data:
            try:
                totalTime = 0
                itemFuture = item.item_future
                input_data = itemFuture[item.input_names[0]]
                use_timestamps = itemFuture[item.input_names[1]]
                children = []
                i = -1
                oldTime = time.time()
                for frame_index, frame in preprocess_video(input_data, self.frame_interval, self.image_size, self.use_half_precision, self.device, use_timestamps):
                    i += 1
                    newTime = time.time()
                    totalTime += newTime - oldTime
                    oldTime = newTime
                    data = {item.output_names[1]: frame, item.output_names[2]: frame_index}
                    result = await ItemFuture.create(item, data, item.item_future.handler)
                    children.append(result)
                print("Preprocessed ", i, " frames in ", totalTime, " seconds at an average of ", totalTime/i, " seconds per frame.")
                await itemFuture.set_data(item.output_names[0], children)
            except FileNotFoundError as fnf_error:
                print(f"File not found error: {fnf_error}")
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                print(f"IO error (video might be corrupted): {io_error}")
                itemFuture.set_exception(io_error)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                itemFuture.set_exception(e)