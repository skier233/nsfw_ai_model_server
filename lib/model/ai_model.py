
import logging
import time
import torch
import torch.nn.functional as F
from typing import Iterable, List
from lib.model.model import Model
from lib.model.ai_model_python.python_model import PythonModel
from lib.utils.torch_device_selector import get_device_string


DEFAULT_MODEL_CAPABILITY = "tagging"
DEFAULT_SUPPORTED_TARGET_SCOPES = ["asset", "frame", "region"]
VALID_MODEL_CAPABILITIES = {"tagging", "detection", "embedding"}
VALID_TARGET_SCOPES = {"asset", "frame", "region"}

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
        self.fill_to_batch = configValues.get("fill_to_batch_size", True)
        self.model_image_size = configValues.get("model_image_size", None)
        self.model_category = _normalize_string_list(configValues.get("model_category", None))
        self.model_type = configValues.get("model_type", "ImClass")
        self.model_version = configValues.get("model_version", None)
        self.model_identifier = configValues.get("model_identifier", None)
        self.category_mappings = configValues.get("category_mappings", None)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.model_capabilities = _resolve_model_capabilities(configValues, self.model_type)
        self.supported_target_scopes = _resolve_supported_target_scopes(configValues)
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for models of type model")
        if self.model_category is not None and len(self.model_category) > 1:
            if self.category_mappings is None:
                raise ValueError("category_mappings is required for models with more than one category")
        self.model = None
        if self.device is None:
            self.device = get_device_string()
        self.localdevice = torch.device(self.device)
        self.logger.debug(f"Using device: {self.device}")
    
        if self.device in ["cpu", "mps"]:
            self.batch_size_per_VRAM_GB = None
            self.max_queue_size = None
            self.max_batch_size = 1
            self.max_model_batch_size = 1


        self.update_batch_with_mutli_models(1)
    
    def update_batch_with_mutli_models(self, model_count):
        batch_multipliers = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        if self.batch_size_per_VRAM_GB is not None and (torch.cuda.is_available() or torch.xpu.is_available()):
            batch_size_temp = self.batch_size_per_VRAM_GB * batch_multipliers[model_count - 1]
            if self.device == "cuda":
                gpuMemory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            elif self.device == "xpu":
                gpuMemory = torch.xpu.get_device_properties(0).total_memory / (1024 ** 3)
            scaledBatchSize = custom_round(batch_size_temp * gpuMemory)
            self.max_model_batch_size = scaledBatchSize
            self.max_batch_size = scaledBatchSize
            self.max_queue_size = scaledBatchSize
            self.logger.debug(f"Setting batch size to {scaledBatchSize} based on VRAM size of {gpuMemory} GB for model {self.model_file_name} ({self.model_category})")

    async def worker_function(self, data):
        try:
            if self.model is None:
                await self.load()
            if self.model is None:
                raise RuntimeError(f"Failed to initialize model runner for {self.model_file_name}")

            resized_images = []
            for item in data:
                itemFuture = item.item_future
                image_tensor = _get_model_input_tensor(itemFuture, item.input_names[0], self.model_image_size)
                resized_images.append(image_tensor)

            images = torch.stack(resized_images, dim=0).to(self.localdevice)

            curr = time.time()
            results = self.model.process_images(images)
            self.logger.debug(f"Processed {len(images)} images in {time.time() - curr} in {self.model_file_name} ({self.model_category})")

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
        if self.model is None:
            self.logger.info(f"Loading model {self.model_file_name} with batch size {self.max_model_batch_size}, {self.max_queue_size}, {self.max_batch_size}")
            if self.model_license_name is None:
                self.model = PythonModel(f"./models/{self.model_file_name}.pt", self.max_model_batch_size, self.device, self.fill_to_batch)
            else:
                from ai_processing import ModelRunner
                model_file_path = f"./models/{self.model_file_name}.pt2.enc"
                if self.model_license_name.endswith(".0"):
                    model_file_path = f"./models/{self.model_file_name}.pt.enc"
                self.model = ModelRunner(model_file_path, f"./models/{self.model_license_name}.lic", self.max_model_batch_size, self.device)
            self.tags = get_index_to_tag_mapping(f"./models/{self.model_file_name}.tags.txt")
            if self.model_category is not None and len(self.model_category) == 1 and self.category_mappings is None:
                self.category_mappings = {i: 0 for i, _ in  enumerate(self.tags)}
        else:
            self.model.load_model()


def _resize_tensor_to_model_size(image_tensor: torch.Tensor, model_image_size) -> torch.Tensor:
    if not isinstance(image_tensor, torch.Tensor):
        return image_tensor

    if model_image_size is None:
        return image_tensor

    target_size = int(model_image_size)
    if target_size <= 0:
        return image_tensor

    if image_tensor.dim() != 3:
        return image_tensor

    current_h = int(image_tensor.shape[-2])
    current_w = int(image_tensor.shape[-1])
    if current_h == target_size and current_w == target_size:
        return image_tensor

    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.to(dtype=image_tensor.dtype)


def _get_model_input_tensor(item_future, input_name: str, model_image_size):
    image_tensor = item_future[input_name]
    if not isinstance(image_tensor, torch.Tensor):
        return image_tensor

    target_size = _normalize_target_size(model_image_size)
    if target_size is None or image_tensor.dim() != 3:
        return image_tensor

    current_h = int(image_tensor.shape[-2])
    current_w = int(image_tensor.shape[-1])
    if current_h == target_size and current_w == target_size:
        return image_tensor

    cache = item_future.data.setdefault("_resized_tensor_cache", {})
    cache_key = (input_name, target_size, current_h, current_w, str(image_tensor.dtype))
    cached_tensor = cache.get(cache_key, None)
    if isinstance(cached_tensor, torch.Tensor):
        return cached_tensor

    resized_tensor = _resize_tensor_to_model_size(image_tensor, target_size)
    cache[cache_key] = resized_tensor
    return resized_tensor


def _normalize_target_size(model_image_size):
    if model_image_size is None:
        return None
    target_size = int(model_image_size)
    if target_size <= 0:
        return None
    return target_size

def _normalize_string_list(raw_value) -> List[str] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, Iterable):
        normalized = []
        for item in raw_value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized or None
    return [str(raw_value)]


def _resolve_model_capabilities(config_values, model_type: str) -> List[str]:
    configured = config_values.get("model_capabilities", config_values.get("capabilities", None))
    normalized = _normalize_string_list(configured)
    if normalized:
        valid = [capability for capability in normalized if capability in VALID_MODEL_CAPABILITIES]
        if valid:
            return valid

    model_type_normalized = (model_type or "").lower()
    if "embed" in model_type_normalized:
        return ["embedding"]
    if "detect" in model_type_normalized:
        return ["detection"]
    return [DEFAULT_MODEL_CAPABILITY]


def _resolve_supported_target_scopes(config_values) -> List[str]:
    configured = config_values.get("supported_target_scopes", config_values.get("target_scopes", None))
    normalized = _normalize_string_list(configured)
    if normalized:
        valid = [scope for scope in normalized if scope in VALID_TARGET_SCOPES]
        if valid:
            return valid
    return DEFAULT_SUPPORTED_TARGET_SCOPES.copy()

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
    if value < 8:
        return int(value)
    # Calculate the remainder when the value is divided by 8
    remainder = int(value) % 8
    # If the remainder is less than or equal to 4, round down
    if remainder <= 5:
        return int(value) - remainder
    # Otherwise, round up
    else:
        return int(value) + (8 - remainder)
