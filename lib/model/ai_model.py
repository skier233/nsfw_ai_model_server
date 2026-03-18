
import logging
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Iterable, List
from lib.model.model import Model
from lib.model.ai_model_python.python_model import PythonModel
from lib.utils.torch_device_selector import get_device_string


DEFAULT_MODEL_CAPABILITY = "tagging"
DEFAULT_SUPPORTED_TARGET_SCOPES = ["asset", "frame", "region"]
VALID_MODEL_CAPABILITIES = {"tagging", "detection", "embedding"}
VALID_TARGET_SCOPES = {"asset", "frame", "region"}

class AIModel(Model):
    """Base class for all AI models (tagging, detection, embedding).

    Owns shared lifecycle: config parsing, device selection, batch sizing,
    model loading (encrypted via ModelRunner or unencrypted via PythonModel),
    and inference dispatch.

    Subclasses implement worker_function() with domain-specific logic.
    """

    def __init__(self, configValues, keep_on_device=False):
        Model.__init__(self, configValues)
        self.max_model_batch_size = configValues.get("max_model_batch_size", 12)
        self.batch_size_per_VRAM_GB = configValues.get("batch_size_per_VRAM_GB", None)
        self.model_file_name = configValues.get("model_file_name", None)
        self.model_license_name = configValues.get("model_license_name", None)
        self.device = configValues.get("device", None)
        self.fill_to_batch = configValues.get("fill_to_batch_size", True)
        self.keep_on_device = keep_on_device
        self.model_image_size = configValues.get("model_image_size", None)
        self.model_category = _normalize_string_list(configValues.get("model_category", None))
        self.model_type = configValues.get("model_type", "ImClass")
        self.model_version = configValues.get("model_version", None)
        self.model_identifier = configValues.get("model_identifier", None)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.model_capabilities = _resolve_model_capabilities(configValues, self.model_type)
        self.supported_target_scopes = _resolve_supported_target_scopes(configValues)
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for AI models")
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

    async def load(self):
        if self.model is not None:
            return
        self.logger.info(f"Loading model {self.model_file_name} with batch size {self.max_model_batch_size}, {self.max_queue_size}, {self.max_batch_size}")
        if self.model_license_name is not None:
            from ai_processing import ModelRunner
            model_file_path = f"./models/{self.model_file_name}.pt2.enc"
            if self.model_license_name.endswith(".0"):
                model_file_path = f"./models/{self.model_file_name}.pt.enc"
            self.model = ModelRunner(model_file_path, f"./models/{self.model_license_name}.lic",
                                     self.max_model_batch_size, self.device,
                                     self.fill_to_batch, self.keep_on_device)
        else:
            pt2 = Path(f"./models/{self.model_file_name}.pt2")
            pt = Path(f"./models/{self.model_file_name}.pt")
            if pt2.exists():
                model_path = str(pt2)
            elif pt.exists():
                model_path = str(pt)
            else:
                raise FileNotFoundError(f"No model file found for {self.model_file_name} (.pt2 or .pt)")
            self.model = PythonModel(model_path, self.max_model_batch_size,
                                      self.device, self.fill_to_batch,
                                      self.keep_on_device)

    def run_inference(self, input_tensor, apply_sigmoid=False):
        """Run model inference on input tensor. Returns output tensor."""
        if apply_sigmoid:
            return self.model.process_images(input_tensor)
        return self.model.run_raw(input_tensor)

    async def worker_function(self, data):
        raise NotImplementedError("Subclasses must implement worker_function")


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
