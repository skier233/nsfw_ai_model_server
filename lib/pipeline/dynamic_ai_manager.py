import logging
import os
from typing import Iterable, List, Optional
from lib.config.config_utils import load_config
from lib.model.ai_model import AIModel
from lib.pipeline.pipeline import ModelWrapper
from lib.migrations.migration_v20 import migrate_to_2_0
from lib.server.exceptions import NoActiveModelsException


ai_active_directory = "./config/active_ai.yaml"



class DynamicAIManager:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.configfile = load_config(ai_active_directory)
        self.ai_model_names = self.configfile.get("active_ai_models", [])
        self.active_capabilities = self.configfile.get("active_capabilities", None)
        self.capability_model_groups = self.configfile.get("capability_model_groups", None)
        self.loaded = False
        self.image_size = None
        self.normalization_config = None
        self.models = []
        self.logger = logging.getLogger("logger")

    def load(self):
        if self.loaded:
            return
        models = []
        self.ai_model_names = self._resolve_active_model_names()

        if self.ai_model_names is None or len(self.ai_model_names) == 0:
            self.logger.error("Error: No active AI models found in active_ai.yaml")
            raise NoActiveModelsException("Error: No active AI models found in active_ai.yaml")
        for model_name in self.ai_model_names:
            models.append(self.model_manager.get_or_create_model(model_name))
        models = [model for model in models if model is not None]
        if len(models) == 0:
            self.logger.error("Error: Active model names resolved, but none could be loaded successfully.")
            raise NoActiveModelsException("Error: Could not load any resolved active AI models")
        self.__verify_models(models)
        self.models = models
        self.loaded = True

    def get_dynamic_models(self, mode, inputs, outputs, required_capabilities=None, required_scope=None):
        self.load()
        selected_models = self._select_models(required_capabilities=required_capabilities, required_scope=required_scope)
        if len(selected_models) == 0:
            requested_caps = _normalize_string_list(required_capabilities)
            requested_scope = required_scope or "any"
            raise ValueError(
                f"Error: No active AI models matched dynamic expansion filters "
                f"(capabilities={requested_caps or ['any']}, scope={requested_scope})"
            )

        normalized_mode = (mode or "").lower()
        if normalized_mode == "video":
            return self._create_dynamic_video_wrappers(inputs, outputs, selected_models)
        if normalized_mode == "image":
            return self._create_dynamic_image_wrappers(inputs, outputs, selected_models)
        if normalized_mode == "region":
            return self._create_dynamic_region_wrappers(inputs, outputs, selected_models)

        raise ValueError(f"Error: Unsupported dynamic mode '{mode}'. Expected 'image', 'video', or 'region'.")

    def get_dynamic_models_from_config(self, model_config):
        mode = model_config.get("dynamic_mode", None)
        if mode is None:
            raise ValueError("Error: dynamic_ai model config requires dynamic_mode set to 'image', 'video', or 'region'.")

        return self.get_dynamic_models(
            mode=mode,
            inputs=model_config["inputs"],
            outputs=model_config["outputs"],
            required_capabilities=model_config.get("required_capabilities", model_config.get("capabilities", None)),
            required_scope=model_config.get("required_scope", model_config.get("target_scope", None)),
        )

    def get_dynamic_video_ai_models(self, inputs, outputs):
        return self.get_dynamic_models(mode="video", inputs=inputs, outputs=outputs)

    def get_dynamic_region_ai_models(self, inputs, outputs, required_capabilities=None):
        return self.get_dynamic_models(
            mode="region",
            inputs=inputs,
            outputs=outputs,
            required_capabilities=required_capabilities,
            required_scope="region",
        )

    def _create_dynamic_video_wrappers(self, inputs, outputs, models):
        model_wrappers = []
        video_preprocessor = self.model_manager.get_or_create_model("video_preprocessor_dynamic")
        image_size, norm_config = self._resolve_preprocess_settings(models)
        video_preprocessor.model.image_size = image_size
        video_preprocessor.model.normalization_config = norm_config

        # add a preprocessor
        model_wrappers.append(ModelWrapper(video_preprocessor, inputs, ["dynamic_children", "dynamic_frame", "frame_index", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"]))

        # add the ai models
        for model in models:
            model_wrappers.append(ModelWrapper(model, ["dynamic_frame", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], model.model.model_category))

        # coalesce all the ai results
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.insert(0, "frame_index")
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, ["dynamic_result"]))

        # finish results
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_finisher"), ["dynamic_result"], []))

        # await children
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("batch_awaiter"), ["dynamic_children"], outputs))
        self.logger.debug("Finished creating dynamic Video AI models")
        return model_wrappers
    
    def get_dynamic_image_ai_models(self, inputs, outputs):
        return self.get_dynamic_models(mode="image", inputs=inputs, outputs=outputs)

    def _create_dynamic_image_wrappers(self, inputs, outputs, models):
        model_wrappers = []
        image_preprocessor = self.model_manager.get_or_create_model("image_preprocessor_dynamic")
        image_size, norm_config = self._resolve_preprocess_settings(models)
        image_preprocessor.model.image_size = image_size
        image_preprocessor.model.normalization_config = norm_config

        # add a preprocessor
        model_wrappers.append(ModelWrapper(image_preprocessor, [inputs[0]], ["dynamic_image"]))

        # add the ai models
        for model in models:
            model_wrappers.append(ModelWrapper(model, ["dynamic_image", inputs[1], inputs[2], inputs[3]], model.model.model_category))

        # coalesce all the ai results
        coalesce_inputs = self._build_coalesce_inputs(models)
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, outputs=outputs))

        self.logger.debug("Finished creating dynamic Image AI models")
        return model_wrappers

    def _create_dynamic_region_wrappers(self, inputs, outputs, models):
        model_wrappers = []

        # Build region children futures from a parent image/frame tensor and region targets.
        # inputs expected: [source_tensor, region_targets, threshold, return_confidence, skipped_categories]
        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("region_children_builder"),
                inputs,
                [
                    "dynamic_region_children",
                    "dynamic_region_image",
                    "dynamic_region_target",
                    "dynamic_threshold",
                    "dynamic_return_confidence",
                    "dynamic_skipped_categories",
                ],
            )
        )

        # Run selected models on each region child image.
        for model in models:
            model_wrappers.append(
                ModelWrapper(
                    model,
                    ["dynamic_region_image", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"],
                    model.model.model_category,
                )
            )

        # Coalesce per-region results and keep region target metadata attached.
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.insert(0, "dynamic_region_target")
        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("result_coalescer"),
                coalesce_inputs,
                ["dynamic_region_result"],
            )
        )

        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("result_finisher"),
                ["dynamic_region_result"],
                [],
            )
        )

        # Await all region children to return array of per-region outputs.
        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("batch_awaiter"),
                ["dynamic_region_children"],
                outputs,
            )
        )

        self.logger.debug("Finished creating dynamic Region AI models")
        return model_wrappers

    def _select_models(self, required_capabilities=None, required_scope=None):
        normalized_capabilities = set(_normalize_string_list(required_capabilities) or [])
        normalized_scope = str(required_scope).strip() if required_scope is not None else None
        selected = []

        for model in self.models:
            model_capabilities = set(getattr(model.model, "model_capabilities", []) or [])
            model_scopes = set(getattr(model.model, "supported_target_scopes", []) or [])

            if normalized_capabilities and not (normalized_capabilities & model_capabilities):
                continue
            if normalized_scope is not None and normalized_scope not in model_scopes:
                continue
            selected.append(model)

        return selected

    def _build_coalesce_inputs(self, models):
        coalesce_inputs = []
        for model in models:
            categories = model.model.model_category
            if isinstance(categories, list):
                coalesce_inputs.extend(categories)
            else:
                coalesce_inputs.append(categories)
        return coalesce_inputs

    def _resolve_preprocess_settings(self, models):
        fallback_size = self.image_size or 512
        fallback_norm = self.normalization_config or 1
        if not models:
            return fallback_size, fallback_norm

        first_model = models[0].model
        image_size = getattr(first_model, "model_image_size", None) or fallback_size
        norm_config = getattr(first_model, "normalization_config", None) or fallback_norm
        return image_size, norm_config
    
    def __verify_models(self, models, second_pass=False):
        first_image_size = None
        first_norm_config = None
        for model in models:
            inner_model = model.model
            if not isinstance(inner_model, AIModel):
                raise ValueError(f"Error: Dynamic AI models must all be AI models! {inner_model} is not an AI model!")
            try:
                if inner_model.model_category is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_category set! {inner_model} does not have model_category set and needs a migration")
                if inner_model.model_version is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_version set! {inner_model} does not have model_version set and needs a migration")
                if inner_model.model_image_size is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_image_size set! {inner_model} does not have model_image_size set and needs a migration")
            except ValueError as e:
                if second_pass:
                    raise e
                self.logger.warning(f"Detected old model files. Attempting migration...")
                migrate_to_2_0()
                self.logger.info("Migration complete. Verifying models again...")
                models = []
                for model_name in self.ai_model_names:
                    models.append(self.model_manager.get_and_refresh_model(model_name))
                self.__verify_models(models, True)
                return

            if first_image_size is None:
                first_image_size = inner_model.model_image_size
            if first_norm_config is None:
                first_norm_config = inner_model.normalization_config
        self.image_size = first_image_size
        self.normalization_config = first_norm_config
        self.logger.debug("Finished verifying dynamic AI models")

    def _resolve_active_model_names(self):
        explicit_names = _normalize_string_list(self.configfile.get("active_ai_models", [])) or []
        capability_flags = self.configfile.get("active_capabilities", None)
        capability_groups = self.configfile.get("capability_model_groups", None)

        if capability_flags is None:
            return _dedupe_strings(explicit_names)

        if not isinstance(capability_flags, dict):
            self.logger.warning("active_capabilities must be a mapping of capability->bool; ignoring malformed value.")
            return _dedupe_strings(explicit_names)

        enabled_capabilities = []
        for capability_name, enabled in capability_flags.items():
            if capability_name not in ["tagging", "detection", "embedding"]:
                self.logger.warning(f"Unknown active_capabilities key '{capability_name}' ignored.")
                continue
            if bool(enabled):
                enabled_capabilities.append(capability_name)

        selected_names = list(explicit_names)
        if enabled_capabilities:
            if isinstance(capability_groups, dict) and capability_groups:
                for capability_name in enabled_capabilities:
                    group_models = _normalize_string_list(capability_groups.get(capability_name, [])) or []
                    selected_names.extend(group_models)
            else:
                self.logger.info("active_capabilities enabled without capability_model_groups; using auto-discovered model capability index.")
                discovered_index = self._build_capability_model_index()
                for capability_name in enabled_capabilities:
                    selected_names.extend(discovered_index.get(capability_name, []))

        deduped = _dedupe_strings(selected_names)
        self.logger.info(
            f"Resolved active AI models: {len(deduped)} model(s) from explicit list + capability flags {enabled_capabilities}"
        )
        return deduped

    def _build_capability_model_index(self):
        capability_index = {"tagging": [], "detection": [], "embedding": []}
        models_directory = "./config/models"

        if not os.path.isdir(models_directory):
            self.logger.warning("Model config directory ./config/models not found while building capability index.")
            return capability_index

        for file_name in os.listdir(models_directory):
            if not file_name.endswith(".yaml"):
                continue

            model_name = file_name[:-5]
            model_config = load_config(os.path.join(models_directory, file_name), default_config={}) or {}
            if model_config.get("type") != "model":
                continue

            capabilities = self._infer_model_capabilities(model_config)
            for capability in capabilities:
                if capability in capability_index:
                    capability_index[capability].append(model_name)

        for capability, names in capability_index.items():
            capability_index[capability] = _dedupe_strings(names)

        return capability_index

    def _infer_model_capabilities(self, model_config):
        configured = model_config.get("model_capabilities", model_config.get("capabilities", None))
        normalized = _normalize_string_list(configured)
        if normalized:
            return [value for value in normalized if value in ["tagging", "detection", "embedding"]] or ["tagging"]

        model_type = str(model_config.get("model_type", "")).lower()
        if "detect" in model_type:
            return ["detection"]
        if "embed" in model_type:
            return ["embedding"]
        return ["tagging"]


def _normalize_string_list(value: Optional[Iterable[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    normalized = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized or None


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped

        

