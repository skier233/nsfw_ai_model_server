import logging
import os
import re
from typing import Iterable, List, Optional
from lib.config.config_utils import load_config
from lib.config.model_capabilities import ModelCapabilitiesConfig
from lib.configurator.configure_model_capabilities import load_model_capabilities_config
from lib.model.ai_model import AIModel
from lib.pipeline.pipeline import ModelWrapper
from lib.migrations.migration_v20 import migrate_to_2_0
from lib.server.exceptions import NoActiveModelsException


ai_active_directory = "./config/active_ai.yaml"



class DynamicAIManager:
    def __init__(self, model_manager):
        self.logger = logging.getLogger("logger")
        self.model_manager = model_manager
        self.configfile = load_config(ai_active_directory)
        self.ai_model_names = self.configfile.get("active_ai_models", [])
        self.active_capabilities = self.configfile.get("active_capabilities", None)
        self.capability_model_groups = self.configfile.get("capability_model_groups", None)
        model_capabilities_config = load_model_capabilities_config()
        self.model_capabilities = ModelCapabilitiesConfig(
            config=model_capabilities_config,
            active_model_library=self.ai_model_names,
            logger=self.logger,
        )
        self.loaded = False
        self.image_size = None
        self.normalization_config = None
        self.models = []
        self.models_by_config_name = {}

    def set_known_pipelines(self, pipeline_names):
        self.model_capabilities.set_known_pipelines(pipeline_names)
        self.model_capabilities.validate()

    def load(self):
        if self.loaded:
            return
        models = []
        self.models_by_config_name = {}
        self.ai_model_names = self._resolve_active_model_names()

        if self.ai_model_names is None or len(self.ai_model_names) == 0:
            self.logger.error("Error: No active AI models found in active_ai.yaml")
            raise NoActiveModelsException("Error: No active AI models found in active_ai.yaml")
        for model_name in self.ai_model_names:
            loaded_model = self.model_manager.get_or_create_model(model_name)
            self.models_by_config_name[model_name] = loaded_model
            models.append(loaded_model)
        models = [model for model in models if model is not None]
        if len(models) == 0:
            self.logger.error("Error: Active model names resolved, but none could be loaded successfully.")
            raise NoActiveModelsException("Error: Could not load any resolved active AI models")
        self.__verify_models(models)
        self.models = models
        self.loaded = True

    def get_dynamic_models(self, mode, inputs, outputs, required_capabilities=None, required_scope=None, pipeline_name=None, model_config=None):
        self.load()
        selected_models = self._select_models(
            required_capabilities=required_capabilities,
            required_scope=required_scope,
            mode=mode,
            pipeline_name=pipeline_name,
            model_config=model_config,
        )
        if len(selected_models) == 0:
            requested_caps = _normalize_string_list(required_capabilities)
            requested_scope = required_scope or "any"
            raise ValueError(
                f"Error: No active AI models matched dynamic expansion filters "
                f"(capabilities={requested_caps or ['any']}, scope={requested_scope})"
            )

        normalized_mode = (mode or "").lower()
        if normalized_mode == "video":
            should_expand_region_branch = not bool(_normalize_string_list(required_capabilities)) and required_scope is None
            return self._create_dynamic_video_wrappers(
                inputs,
                outputs,
                selected_models,
                pipeline_name=pipeline_name,
                enable_region_branch=should_expand_region_branch,
            )
        if normalized_mode == "image":
            should_expand_region_branch = not bool(_normalize_string_list(required_capabilities)) and required_scope is None
            return self._create_dynamic_image_wrappers(
                inputs,
                outputs,
                selected_models,
                pipeline_name=pipeline_name,
                enable_region_branch=should_expand_region_branch,
            )
        if normalized_mode == "region":
            return self._create_dynamic_region_wrappers(inputs, outputs, selected_models)

        raise ValueError(f"Error: Unsupported dynamic mode '{mode}'. Expected 'image', 'video', or 'region'.")

    def get_dynamic_models_from_config(self, model_config, pipeline_name=None):
        mode = model_config.get("dynamic_mode", None)
        if mode is None:
            raise ValueError("Error: dynamic_ai model config requires dynamic_mode set to 'image', 'video', or 'region'.")

        return self.get_dynamic_models(
            mode=mode,
            inputs=model_config["inputs"],
            outputs=model_config["outputs"],
            required_capabilities=model_config.get("required_capabilities", model_config.get("capabilities", None)),
            required_scope=model_config.get("required_scope", model_config.get("target_scope", None)),
            pipeline_name=pipeline_name,
            model_config=model_config,
        )

    def get_dynamic_video_ai_models(self, inputs, outputs, pipeline_name=None):
        return self.get_dynamic_models(mode="video", inputs=inputs, outputs=outputs, pipeline_name=pipeline_name)

    def get_dynamic_region_ai_models(self, inputs, outputs, required_capabilities=None, pipeline_name=None):
        return self.get_dynamic_models(
            mode="region",
            inputs=inputs,
            outputs=outputs,
            required_capabilities=required_capabilities,
            required_scope="region",
            pipeline_name=pipeline_name,
        )

    def _create_dynamic_video_wrappers(self, inputs, outputs, models, pipeline_name=None, enable_region_branch=False):
        model_wrappers = []
        video_preprocessor = self.model_manager.get_or_create_model("video_preprocessor_dynamic")
        image_size, norm_config = self._resolve_preprocess_settings(models)
        video_preprocessor.model.image_size = image_size
        video_preprocessor.model.normalization_config = norm_config

        # Pre-scan detectors: if any detector has region branches, add a
        # single extra output for the original-resolution raw frame.  Both
        # detectors (which handle their own internal 640-resize with correct
        # aspect ratio) and region builders (which crop/align subimages at
        # full resolution) share this tensor.
        _region_frame_key = None   # set once when first detector is found
        _extra_frame_specs = []
        preproc_outputs = [
            "dynamic_children", "dynamic_frame", "frame_index",
            "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories",
        ]

        if pipeline_name and enable_region_branch:
            _prescan_detector_names = self.model_capabilities.resolve_model_names_for_stage(
                pipeline_name=pipeline_name,
                stage="detector",
                available_models=self.models,
            )
            if _prescan_detector_names:
                # One original-resolution raw frame output shared by all
                # detectors and region branches.
                _region_frame_key = "dynamic_frame__region_source"
                out_idx = len(preproc_outputs)
                preproc_outputs.append(_region_frame_key)
                _extra_frame_specs.append({
                    "output_index": out_idx,
                    "image_size": 0,     # no resize — keep original frame resolution
                    "norm_config": -1,   # raw [0-255] pixels
                    "use_half": False,
                })

        video_preprocessor.model.extra_frame_specs = _extra_frame_specs
        model_wrappers.append(ModelWrapper(video_preprocessor, inputs, preproc_outputs))

        # add the ai models
        for model in models:
            model_wrappers.append(ModelWrapper(model, ["dynamic_frame", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], model.model.model_category))

        additional_coalesce_inputs = []
        if pipeline_name and enable_region_branch:
            detector_names = self.model_capabilities.resolve_model_names_for_stage(
                pipeline_name=pipeline_name,
                stage="detector",
                available_models=self.models,
            )
            region_model_rules = self.model_capabilities.get_region_model_rules(pipeline_name)

            if detector_names and region_model_rules:
                detector_models = self._select_models_by_name(detector_names)
                region_rules_by_detector = {
                    str(rule.get("key", "")).strip(): rule
                    for rule in region_model_rules
                }

                for detector_name, detector_model in zip(detector_names, detector_models):
                    detector_outputs = _normalize_string_list(detector_model.model.model_category) or []
                    additional_coalesce_inputs.extend(detector_outputs)

                    # All detectors receive the original-resolution raw frame.
                    # run_detection() inside each model handles its own resize
                    # to the model’s det_size (e.g. 640×640) with correct
                    # aspect-ratio preservation and produces bboxes/kps in
                    # original-frame coordinates.
                    detector_frame_key = _region_frame_key or "dynamic_frame"

                    model_wrappers.append(
                        ModelWrapper(
                            detector_model,
                            [detector_frame_key, "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"],
                            detector_outputs,
                        )
                    )

                    detector_rule = region_rules_by_detector.get(detector_name, None)
                    if detector_rule is None:
                        continue

                    region_model_names = list(detector_rule.get("models", []) or [])
                    if not region_model_names:
                        label_model_map = detector_rule.get("label_models", {}) or {}
                        for label_models in label_model_map.values():
                            region_model_names.extend(_normalize_string_list(label_models) or [])
                        region_model_names = _dedupe_strings(region_model_names)
                        if region_model_names:
                            self.logger.warning(
                                f"Pipeline '{pipeline_name}' defines label-specific region_models for detector '{detector_name}', "
                                f"but label-specific routing is not available yet. Running union of all configured label model lists."
                            )

                    if not region_model_names:
                        continue

                    if len(detector_outputs) == 0:
                        self.logger.warning(
                            f"Skipping region branch for detector '{detector_name}' in pipeline '{pipeline_name}' because it emits no output categories"
                        )
                        continue

                    detector_output_key = detector_outputs[0]
                    if len(detector_outputs) > 1:
                        self.logger.warning(
                            f"Detector '{detector_name}' in pipeline '{pipeline_name}' emits multiple output keys {detector_outputs}. "
                            f"Using '{detector_output_key}' for region target extraction."
                        )

                    alias = _sanitize_key(detector_name)
                    region_targets_key = f"region_targets__{alias}"
                    region_errors_key = f"region_errors__{alias}"
                    children_key = f"dynamic_region_children__{alias}"
                    region_image_key = f"dynamic_region_image__{alias}"
                    region_target_key = f"dynamic_region_target__{alias}"
                    threshold_key = f"dynamic_threshold__{alias}"
                    return_confidence_key = f"dynamic_return_confidence__{alias}"
                    skipped_categories_key = f"dynamic_skipped_categories__{alias}"
                    region_result_key = f"dynamic_region_result__{alias}"
                    region_results_key = f"dynamic_region_results__{alias}"

                    model_wrappers.append(
                        ModelWrapper(
                            self.model_manager.get_or_create_model("detector_result_to_region_targets"),
                            [detector_output_key, "frame_index", detector_frame_key, "frame_index"],
                            [region_targets_key, region_errors_key],
                        )
                    )

                    # Region source is the original-resolution raw frame.
                    model_wrappers.append(
                        ModelWrapper(
                            self.model_manager.get_or_create_model("region_children_builder"),
                            [detector_frame_key, region_targets_key, "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"],
                            [
                                children_key,
                                region_image_key,
                                region_target_key,
                                threshold_key,
                                return_confidence_key,
                                skipped_categories_key,
                                f"dynamic_region_source__{alias}",
                            ],
                        )
                    )

                    region_models = self._select_models_by_name(region_model_names)
                    region_branch_outputs = []
                    for region_model in region_models:
                        branch_outputs = _normalize_string_list(region_model.model.model_category) or []
                        region_branch_outputs.extend(branch_outputs)
                        model_wrappers.append(
                            ModelWrapper(
                                region_model,
                                [region_image_key, threshold_key, return_confidence_key, skipped_categories_key],
                                branch_outputs,
                            )
                        )

                    model_wrappers.append(
                        ModelWrapper(
                            self.model_manager.get_or_create_model("result_coalescer"),
                            [region_target_key] + region_branch_outputs,
                            [region_result_key],
                        )
                    )

                    model_wrappers.append(
                        ModelWrapper(
                            self.model_manager.get_or_create_model("result_finisher"),
                            [region_result_key],
                            [],
                        )
                    )

                    model_wrappers.append(
                        ModelWrapper(
                            self.model_manager.get_or_create_model("batch_awaiter"),
                            [children_key],
                            [region_results_key],
                        )
                    )

                    additional_coalesce_inputs.extend([region_targets_key, region_errors_key, region_results_key])

        # coalesce all the ai results
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.extend(additional_coalesce_inputs)
        coalesce_inputs = _dedupe_strings(coalesce_inputs)
        coalesce_inputs.insert(0, "frame_index")
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, ["dynamic_result"]))

        # finish results
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_finisher"), ["dynamic_result"], []))

        # await children
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("batch_awaiter"), ["dynamic_children"], outputs))
        self.logger.debug("Finished creating dynamic Video AI models")
        return model_wrappers
    
    def get_dynamic_image_ai_models(self, inputs, outputs, pipeline_name=None):
        return self.get_dynamic_models(mode="image", inputs=inputs, outputs=outputs, pipeline_name=pipeline_name)

    def _create_dynamic_image_wrappers(self, inputs, outputs, models, pipeline_name=None, enable_region_branch=False):
        model_wrappers = []
        image_preprocessor = self.model_manager.get_or_create_model("image_preprocessor_dynamic")
        image_size, norm_config = self._resolve_preprocess_settings(models)
        image_preprocessor.model.image_size = image_size
        image_preprocessor.model.normalization_config = norm_config

        # ── Peek ahead: do we need a region-source (raw) tensor? ──
        # If detectors + region models are configured we produce the raw
        # original-resolution [0-255] tensor from the SAME preprocessor
        # that builds the classification tensor – one disk read instead of
        # two.  The preprocessor's dual-output mode emits the raw tensor
        # as its second output_name.
        _region_source_key = None
        _needs_region_source = False
        if pipeline_name and enable_region_branch:
            _det_names = self.model_capabilities.resolve_model_names_for_stage(
                pipeline_name=pipeline_name,
                stage="detector",
                available_models=self.models,
            )
            _region_rules = self.model_capabilities.get_region_model_rules(pipeline_name)
            _needs_region_source = bool(_det_names) and bool(_region_rules)

        if _needs_region_source:
            _region_source_key = "dynamic_image__region_source"
            model_wrappers.append(
                ModelWrapper(image_preprocessor, [inputs[0]], ["dynamic_image", _region_source_key])
            )
        else:
            model_wrappers.append(ModelWrapper(image_preprocessor, [inputs[0]], ["dynamic_image"]))

        # add the ai models
        for model in models:
            model_wrappers.append(ModelWrapper(model, ["dynamic_image", inputs[1], inputs[2], inputs[3]], model.model.model_category))

        additional_coalesce_inputs = []
        if _needs_region_source:
            detector_names = _det_names
            region_model_rules = _region_rules
            detector_models = self._select_models_by_name(detector_names)
            region_rules_by_detector = {
                str(rule.get("key", "")).strip(): rule
                for rule in region_model_rules
            }

            for detector_name, detector_model in zip(detector_names, detector_models):
                detector_outputs = _normalize_string_list(detector_model.model.model_category) or []
                additional_coalesce_inputs.extend(detector_outputs)

                # Every detector receives the original-resolution raw
                # image.  run_detection() inside each model handles its
                # own resize to the model's det_size (e.g. 640×640) with
                # correct aspect-ratio preservation and produces bboxes/
                # kps in original-image coordinates.
                model_wrappers.append(
                    ModelWrapper(
                        detector_model,
                        [_region_source_key, inputs[1], inputs[2], inputs[3]],
                        detector_outputs,
                    )
                )

                detector_rule = region_rules_by_detector.get(detector_name, None)
                if detector_rule is None:
                    continue

                region_model_names = list(detector_rule.get("models", []) or [])
                if not region_model_names:
                    label_model_map = detector_rule.get("label_models", {}) or {}
                    for label_models in label_model_map.values():
                        region_model_names.extend(_normalize_string_list(label_models) or [])
                    region_model_names = _dedupe_strings(region_model_names)
                    if region_model_names:
                        self.logger.warning(
                            f"Pipeline '{pipeline_name}' defines label-specific region_models for detector '{detector_name}', "
                            f"but label-specific routing is not available yet. Running union of all configured label model lists."
                        )

                if not region_model_names:
                    continue

                if len(detector_outputs) == 0:
                    self.logger.warning(
                        f"Skipping region branch for detector '{detector_name}' in pipeline '{pipeline_name}' because it emits no output categories"
                    )
                    continue

                detector_output_key = detector_outputs[0]
                if len(detector_outputs) > 1:
                    self.logger.warning(
                        f"Detector '{detector_name}' in pipeline '{pipeline_name}' emits multiple output keys {detector_outputs}. "
                        f"Using '{detector_output_key}' for region target extraction."
                    )

                alias = _sanitize_key(detector_name)
                region_targets_key = f"region_targets__{alias}"
                region_errors_key = f"region_errors__{alias}"
                children_key = f"dynamic_region_children__{alias}"
                region_image_key = f"dynamic_region_image__{alias}"
                region_target_key = f"dynamic_region_target__{alias}"
                threshold_key = f"dynamic_threshold__{alias}"
                return_confidence_key = f"dynamic_return_confidence__{alias}"
                skipped_categories_key = f"dynamic_skipped_categories__{alias}"
                region_result_key = f"dynamic_region_result__{alias}"
                region_results_key = f"face_region_results__{alias}"

                # Region source is the original-resolution image.
                # Detection coords (bboxes, kps) are already in original
                # space (run_detection scales them back), so cropping and
                # ArcFace alignment operate at full resolution.

                model_wrappers.append(
                    ModelWrapper(
                        self.model_manager.get_or_create_model("detector_result_to_region_targets"),
                        [detector_output_key, inputs[0], _region_source_key],
                        [region_targets_key, region_errors_key],
                    )
                )

                model_wrappers.append(
                    ModelWrapper(
                        self.model_manager.get_or_create_model("region_children_builder"),
                        [_region_source_key, region_targets_key, inputs[1], inputs[2], inputs[3]],
                        [
                            children_key,
                            region_image_key,
                            region_target_key,
                            threshold_key,
                            return_confidence_key,
                            skipped_categories_key,
                            f"dynamic_region_source__{alias}",
                        ],
                    )
                )

                region_models = self._select_models_by_name(region_model_names)
                region_branch_outputs = []
                for region_model in region_models:
                    branch_outputs = _normalize_string_list(region_model.model.model_category) or []
                    region_branch_outputs.extend(branch_outputs)
                    model_wrappers.append(
                        ModelWrapper(
                            region_model,
                            [region_image_key, threshold_key, return_confidence_key, skipped_categories_key],
                            branch_outputs,
                        )
                    )

                model_wrappers.append(
                    ModelWrapper(
                        self.model_manager.get_or_create_model("result_coalescer"),
                        [region_target_key] + region_branch_outputs,
                        [region_result_key],
                    )
                )

                model_wrappers.append(
                    ModelWrapper(
                        self.model_manager.get_or_create_model("result_finisher"),
                        [region_result_key],
                        [],
                    )
                )

                model_wrappers.append(
                    ModelWrapper(
                        self.model_manager.get_or_create_model("batch_awaiter"),
                        [children_key],
                        [region_results_key],
                    )
                )

                additional_coalesce_inputs.extend([region_targets_key, region_errors_key, region_results_key])

        # coalesce all the ai results
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.extend(additional_coalesce_inputs)
        coalesce_inputs = _dedupe_strings(coalesce_inputs)
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
                    "dynamic_region_source",
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

    def _select_models(self, required_capabilities=None, required_scope=None, mode=None, pipeline_name=None, model_config=None):
        normalized_capabilities = set(_normalize_string_list(required_capabilities) or [])
        normalized_scope = str(required_scope).strip() if required_scope is not None else None
        candidate_models = self.models

        dynamic_stage = self._infer_dynamic_stage(mode, normalized_capabilities)
        if pipeline_name:
            configured_model_names = self.model_capabilities.resolve_model_names_for_stage(
                pipeline_name=pipeline_name,
                stage=dynamic_stage,
                available_models=self.models,
            )
            if configured_model_names is not None:
                candidate_models = self._select_models_by_name(configured_model_names)

        selected = []

        for model in candidate_models:
            model_capabilities = set(getattr(model.model, "model_capabilities", []) or [])
            model_scopes = set(getattr(model.model, "supported_target_scopes", []) or [])

            if normalized_capabilities and not (normalized_capabilities & model_capabilities):
                continue
            if normalized_scope is not None and normalized_scope not in model_scopes:
                continue
            selected.append(model)

        return selected

    def _infer_dynamic_stage(self, mode, normalized_capabilities):
        if str(mode or "").strip().lower() == "region":
            return "region"
        if "detection" in normalized_capabilities:
            return "detector"
        return "full_image"

    def _select_models_by_name(self, model_names):
        ordered_names = _normalize_string_list(model_names) or []
        if not ordered_names:
            return []

        selected = []
        missing = []
        for model_name in ordered_names:
            model = self.models_by_config_name.get(model_name, None)
            if model is None:
                missing.append(model_name)
                continue
            selected.append(model)

        if missing:
            raise ValueError(
                f"Error: model_capabilities selected model(s) that are not currently loadable from active_ai_models: {missing}"
            )

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
            model_type = model_config.get("type", "")
            if model_type not in ("model", "face_torch_export"):
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


def _sanitize_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text)


def _get_detector_preprocess_spec(detector_model):
    """Return the ``(image_size, normalization_config, use_half_precision)``
    that *detector_model* requires, or ``None`` when it is happy with the
    default pipeline preprocessor output.

    The tuple is used to configure a dedicated preprocessor so the detector
    receives a tensor at exactly its declared resolution and pixel format
    without storing a full-resolution copy.
    """
    inner = getattr(detector_model, "model", None)
    if inner is None:
        return None
    face_role = getattr(inner, "face_model_role", None)
    model_type = str(getattr(inner, "model_type", "") or "").lower().replace("_", "").replace(" ", "")
    if face_role is not None or "facetorchexport" in model_type:
        return (
            getattr(inner, "model_image_size", 640) or 640,
            -1,     # raw [0-255] pixels, no normalization
            False,  # float32 — face models need full precision
        )
    return None


