
import logging
import torch
from lib.model.model import Model
from lib.model.ai_model_python.python_model import PythonModel
import time

class AIModel(Model):
    def __init__(self, configValues):
        Model.__init__(self, configValues)
        self.max_model_batch_size = configValues.get("max_model_batch_size", 12)
        self.batch_size_per_VRAM_GB = configValues.get("batch_size_per_VRAM_GB", None)
        self.model_file_name = configValues.get("model_file_name", None)
        self.model_license_name = configValues.get("model_license_name", None)
        self.model_threshold = configValues.get("model_threshold", None)
        self.model_return_tags = configValues.get("model_return_tags", False)
        self.model_return_confidence = configValues.get("model_return_confidence", False)
        self.device = configValues.get("device", None)
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for models of type model")
        self.model = None
        if self.device is None:
            self.localdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.localdevice = torch.device(self.device)

        if self.batch_size_per_VRAM_GB is not None:
            gpuMemory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            scaledBatchSize = custom_round(self.batch_size_per_VRAM_GB * gpuMemory)
            if self.max_model_batch_size == -1:
                self.max_model_batch_size = scaledBatchSize
                self.max_batch_size = scaledBatchSize
                self.max_queue_size = scaledBatchSize
                self.logger.debug(f"Setting batch size to {scaledBatchSize} based on VRAM size of {gpuMemory} GB")
                
    

    async def worker_function(self, data):
        # Get the shape of the first image in the data
        first_image_shape = data[0].item_future[data[0].input_names[0]].shape

        # Create an empty tensor with the same shape as the input images
        images = torch.empty((len(data), *first_image_shape), device=self.localdevice)
        for i, item in enumerate(data):
            itemFuture = item.item_future
            images[i] = itemFuture[item.input_names[0]]

        curr = time.time()
        results = self.model.process_images(images)
        self.logger.debug(f"Processed {len(images)} images in {time.time() - curr}")

        for i, item in enumerate(data):
            item_future = item.item_future
            threshold = item_future[item.input_names[1]] or self.model_threshold
            return_confidence = self.model_return_confidence
            if item_future[item.input_names[2]] is not None:
                return_confidence = item_future[item.input_names[2]]
            result = results[i]
            if threshold is not None:
                if return_confidence:
                    result = [(self.tags[i], round(confidence.item(), 2)) for i, confidence in enumerate(result) if confidence.item() > threshold]
                else:
                    result = (result > threshold)
                    if self.model_return_tags:
                        result = [self.tags[i] for i, tag in enumerate(result) if tag.item()]
            await item_future.set_data(item.output_names[0], result)

    async def load(self):
        if self.model is None:
            if self.model_license_name is None:
                self.model = PythonModel(f"./models/{self.model_file_name}.pt", self.max_model_batch_size, self.device)
            else:
                from ai_processing import ModelRunner
                self.model = ModelRunner(f"./models/{self.model_file_name}.pt.enc", f"./models/{self.model_license_name}.lic", self.max_model_batch_size, self.device)
            self.tags = get_index_to_tag_mapping(f"./models/{self.model_file_name}.tags.txt")
        else:
            self.model.load_model()

def get_index_to_tag_mapping(path):
    """
    Retrieves a mapping from indices to tag names by reading from a text file.

    Parameters:
    - tags_txt_path: Path to the text file containing tags, one on each line.

    Returns:
    - A dictionary mapping indices to tag names.
    """
    tags_txt_path = path
    index_to_tag = {}
    with open(tags_txt_path, 'r', encoding='utf-8') as file:
        for index, tag in enumerate(file):
            index_to_tag[index] = tag.strip()  # Remove any leading/trailing whitespace
    return index_to_tag

def custom_round(value):
    # Calculate the difference between the value and the next highest integer
    difference = -value % 1
    # If the difference is less than or equal to 0.1, round up
    if difference <= 0.1:
        return int(value) + 1
    # Otherwise, round down
    else:
        return int(value)