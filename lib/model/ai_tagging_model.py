
import time
import torch
from lib.model.ai_model import AIModel


class AITaggingModel(AIModel):
    """AI model subclass for image tagging / multi-label classification."""

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)
        self.model_threshold = configValues.get("model_threshold", None)
        self.model_return_tags = configValues.get("model_return_tags", False)
        self.model_return_confidence = configValues.get("model_return_confidence", False)
        self.category_mappings = configValues.get("category_mappings", None)
        if self.model_category is not None and len(self.model_category) > 1:
            if self.category_mappings is None:
                raise ValueError("category_mappings is required for models with more than one category")

    async def worker_function(self, data):
        try:
            if self.model is None:
                await self.load()
            if self.model is None:
                raise RuntimeError(f"Failed to initialize model runner for {self.model_file_name}")

            resized_images = []
            for item in data:
                resized_images.append(item.item_future[item.input_names[0]])
            images = torch.stack(resized_images, dim=0).to(self.localdevice)
            curr = time.time()
            results = self.model.process_images(images)
            self.logger.debug(f"Processed {len(data)} images in {time.time() - curr} in {self.model_file_name} ({self.model_category})")
            for i, item in enumerate(data):
                item_future = item.item_future
                threshold = item_future[item.input_names[1]] or self.model_threshold
                return_confidence = self.model_return_confidence
                if item_future[item.input_names[2]] is not None:
                    return_confidence = item_future[item.input_names[2]]
                
                toReturn = {output_name: [] for output_name in item.output_names}
                result = results[i]
                
                for j, confidence in enumerate(result):
                    if threshold is not None:
                        if confidence.item() > threshold:
                            tag_name = self.tags[j]
                            if return_confidence:
                                tag = (tag_name, round(confidence.item(), 2))
                            else:
                                tag = tag_name

                            if j in self.category_mappings:
                                list_id = self.category_mappings[j]
                                toReturn[item.output_names[list_id]].append(tag)
                    else:
                        list_id = self.category_mappings[j]
                        toReturn[item.output_names[list_id]].append(tag)
                for output_name, result_list in toReturn.items():
                    await item_future.set_data(output_name, result_list)
        except Exception as e:
            self.logger.error(f"Error in ai model worker_function: {e}")
            self.logger.debug(f"Error in {self.model_file_name} data: {data}")
            self.logger.debug("Stack trace:", exc_info=True)
            for item in data:
                item.item_future.set_exception(e)

    async def load(self):
        await super().load()
        if not hasattr(self, 'tags') or self.tags is None:
            self.tags = _get_index_to_tag_mapping(f"./models/{self.model_file_name}.tags.txt")
            if self.model_category is not None and len(self.model_category) == 1 and self.category_mappings is None:
                self.category_mappings = {i: 0 for i, _ in enumerate(self.tags)}


def _get_index_to_tag_mapping(path):
    """
    Retrieves a mapping from indices to tag names by reading from a text file.

    Parameters:
    - tags_txt_path: Path to the text file containing tags, one on each line.

    Returns:
    - A dictionary mapping indices to tag names.
    """
    index_to_tag = {}
    with open(path, 'r', encoding='utf-8') as file:
        for index, tag in enumerate(file):
            index_to_tag[index] = tag.strip()
    return index_to_tag
