from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


class ModelCapabilitiesConfig:
    def __init__(
        self,
        config: Dict[str, Any],
        active_model_library: Sequence[str],
        logger,
        known_pipelines: Optional[Iterable[str]] = None,
    ):
        self.config = config or {}
        self.active_model_library = set(_normalize_string_list(active_model_library))
        self.known_pipelines = set(_normalize_string_list(known_pipelines))
        self.logger = logger

    def set_known_pipelines(self, pipelines: Optional[Iterable[str]]):
        self.known_pipelines = set(_normalize_string_list(pipelines))

    def validate(self):
        if not isinstance(self.config, dict):
            raise ValueError("model_capabilities config must be a mapping of pipeline_name -> capability settings")

        for pipeline_name, raw_entry in self.config.items():
            pipeline_name = str(pipeline_name).strip()
            if not pipeline_name:
                raise ValueError("model_capabilities contains an empty pipeline name key")
            if self.known_pipelines and pipeline_name not in self.known_pipelines:
                raise ValueError(
                    f"model_capabilities references unknown pipeline '{pipeline_name}'. "
                    f"Known pipelines: {sorted(self.known_pipelines)}"
                )

            if not isinstance(raw_entry, dict):
                raise ValueError(f"model_capabilities entry for pipeline '{pipeline_name}' must be a mapping")

            _ = self.resolve_full_image_model_names_for_validation(pipeline_name)
            detector_models = _normalize_string_list(raw_entry.get("detector_models", []))
            detector_model_set = set(detector_models)
            missing_detectors = [name for name in detector_models if name not in self.active_model_library]
            if missing_detectors:
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' detector_models contains model(s) not in active_ai_models: {missing_detectors}"
                )

            for rule in self.get_region_model_rules(pipeline_name):
                detector_model_name = rule["key"]
                model_names = _normalize_string_list(rule.get("models", []))
                label_models = rule.get("label_models", {}) or {}

                if detector_model_name not in detector_model_set:
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models key '{detector_model_name}' must also be listed in detector_models"
                    )

                missing = [name for name in model_names if name not in self.active_model_library]
                if missing:
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models['{detector_model_name}'] contains model(s) not in active_ai_models: {missing}"
                    )

                for label_name, label_model_names in label_models.items():
                    missing_label_models = [name for name in label_model_names if name not in self.active_model_library]
                    if missing_label_models:
                        raise ValueError(
                            f"model_capabilities pipeline '{pipeline_name}' region_models['{detector_model_name}']['{label_name}'] contains model(s) not in active_ai_models: {missing_label_models}"
                        )

    def resolve_full_image_model_names_for_validation(self, pipeline_name: str) -> List[str]:
        entry = self.config.get(pipeline_name, {}) or {}
        full_image_models = entry.get("full_image_models", "ALL")
        return self._resolve_full_image_model_names(full_image_models, None)

    def get_pipeline_entry(self, pipeline_name: str) -> Dict[str, Any]:
        entry = self.config.get(pipeline_name, None)
        if entry is None:
            return {}
        if not isinstance(entry, dict):
            raise ValueError(f"model_capabilities entry for pipeline '{pipeline_name}' must be a mapping")
        return entry

    def resolve_model_names_for_stage(self, pipeline_name: str, stage: str, available_models: Sequence[Any]) -> Optional[List[str]]:
        entry = self.get_pipeline_entry(pipeline_name)
        if not entry:
            return None

        stage_name = str(stage).strip().lower()
        if stage_name == "detector":
            detectors = _normalize_string_list(entry.get("detector_models", []))
            return detectors if detectors else None

        if stage_name == "region":
            ordered = []
            seen = set()
            for rule in self.get_region_model_rules(pipeline_name):
                model_names = list(rule.get("models", []))
                if not model_names:
                    label_models = rule.get("label_models", {}) or {}
                    for names in label_models.values():
                        model_names.extend(names)

                for model_name in model_names:
                    if model_name in seen:
                        continue
                    seen.add(model_name)
                    ordered.append(model_name)
            return ordered if ordered else None

        if stage_name == "full_image":
            full_image_models = entry.get("full_image_models", "ALL")
            resolved = self._resolve_full_image_model_names(full_image_models, available_models)
            return resolved if resolved else None

        return None

    def get_region_model_rules(self, pipeline_name: str) -> List[Dict[str, Any]]:
        entry = self.get_pipeline_entry(pipeline_name)
        if not entry:
            return []

        region_model_map = entry.get("region_models", {}) or {}
        if not isinstance(region_model_map, dict):
            raise ValueError(
                f"model_capabilities pipeline '{pipeline_name}' region_models must be a mapping"
            )

        rules = []
        for raw_key, raw_value in region_model_map.items():
            detector_model_name = str(raw_key).strip()
            if not detector_model_name:
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models contains an empty key"
                )

            if isinstance(raw_value, dict):
                label_models = {}
                for raw_label, raw_models in raw_value.items():
                    label = str(raw_label).strip()
                    if not label:
                        raise ValueError(
                            f"model_capabilities pipeline '{pipeline_name}' region_models['{detector_model_name}'] contains empty label key"
                        )
                    names = _normalize_string_list(raw_models)
                    if not names:
                        raise ValueError(
                            f"model_capabilities pipeline '{pipeline_name}' region_models['{detector_model_name}']['{label}'] must contain at least one model"
                        )
                    label_models[label] = names

                rules.append({
                    "key": detector_model_name,
                    "models": [],
                    "label_models": label_models,
                })
                continue

            names = _normalize_string_list(raw_value)
            if not names:
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models['{detector_model_name}'] must contain at least one model"
                )
            rules.append({
                "key": detector_model_name,
                "models": names,
                "label_models": {},
            })

        return rules

    def _resolve_full_image_model_names(self, raw_value: Any, available_models: Optional[Sequence[Any]]) -> List[str]:
        if isinstance(raw_value, list):
            names = _normalize_string_list(raw_value)
            missing = [name for name in names if name not in self.active_model_library]
            if missing:
                raise ValueError(
                    f"model_capabilities full_image_models references model(s) not in active_ai_models: {missing}"
                )
            return names

        if isinstance(raw_value, str):
            text = raw_value.strip().upper()
            if text == "ALL":
                return self._resolve_all_tagging_model_names(available_models, exclude_names=[])
            raise ValueError("model_capabilities full_image_models string value must be 'ALL' or explicit list")

        if isinstance(raw_value, dict):
            mode = str(raw_value.get("mode", "")).strip().upper()
            if mode != "ALL":
                raise ValueError("model_capabilities full_image_models mapping mode must be 'ALL'")
            exclude_names = _normalize_string_list(raw_value.get("exclude", []))
            missing_excludes = [name for name in exclude_names if name not in self.active_model_library]
            if missing_excludes:
                raise ValueError(
                    f"model_capabilities full_image_models.exclude references model(s) not in active_ai_models: {missing_excludes}"
                )
            return self._resolve_all_tagging_model_names(available_models, exclude_names=exclude_names)

        raise ValueError(
            "model_capabilities full_image_models must be one of: list, 'ALL', or {mode: 'ALL', exclude: [...]}"
        )

    def _resolve_all_tagging_model_names(self, available_models: Optional[Sequence[Any]], exclude_names: List[str]) -> List[str]:
        exclude_set = set(exclude_names)
        if not available_models:
            return [name for name in sorted(self.active_model_library) if name not in exclude_set]

        resolved = []
        for model in available_models:
            model_name = str(getattr(model.model, "model_file_name", "")).strip()
            if not model_name or model_name in exclude_set:
                continue
            capabilities = set(getattr(model.model, "model_capabilities", []) or [])
            if "tagging" not in capabilities:
                continue
            resolved.append(model_name)
        return resolved


def _normalize_string_list(raw_value) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return [text] if text else []
    normalized = []
    for item in raw_value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized
