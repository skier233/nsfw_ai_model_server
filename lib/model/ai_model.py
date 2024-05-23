
import torch
from ai_processing import ModelRunner
from lib.model.model import Model
import time

class AIModel(Model):
    def __init__(self, configValues):
        Model.__init__(self, configValues)
        self.max_model_batch_size = configValues.get("max_model_batch_size", 12)
        self.model_file_name = configValues.get("model_file_name", None)
        self.model_license_name = configValues.get("model_license_name", None)
        self.model_threshold = configValues.get("model_threshold", None)
        self.model_return_tags = configValues.get("model_return_tags", False)
        self.device = configValues.get("device", None)
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for models of type model")
        if self.model_license_name is None:
            raise ValueError("model_license_name is required for models of type model")
        self.model = None
        if self.device is None:
            self.localdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.localdevice = torch.device(self.device)
    

    async def worker_function(self, data):
        # Get the shape of the first image in the data
        first_image_shape = data[0].item_future[data[0].input_names[0]].shape

        # Create an empty tensor with the same shape as the input images
        images = torch.empty((len(data), *first_image_shape), device=self.localdevice)
        for i, item in enumerate(data):
            itemFuture = item.item_future
            images[i] = itemFuture[item.input_names[0]]
            
        results = self.model.process_images(images)
        #print(f"Processed {len(images)} images in {time.time() - curr}")
        if self.model_threshold is not None:
            results = (results > self.model_threshold)
            if self.model_return_tags:
                results = [[self.tags[i] for i, tag in enumerate(result) if tag.item()] for result in results]

        for item, result in zip(data, results):
            itemFuture = item.item_future
            await itemFuture.set_data(item.output_names[0], result)

    async def load(self):
        if self.model is None:
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