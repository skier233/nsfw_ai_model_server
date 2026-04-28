from typing import Iterable

from lib.configurator.configure_active_ai import (
    load_active_ai_config,
    load_active_ai_models,
    load_available_ai_models,
    load_pinned_ai_models,
    save_active_ai_config,
)


VALID_LOAD_POLICIES = {"use_loaded", "load_if_cheap", "load_or_fail"}


def get_model_catalog(server_manager):
    active_model_names = set(load_active_ai_models())
    pinned_model_names = set(load_pinned_ai_models())
    loaded_model_names = set()
    loaded_model_file_names = set()

    for pipeline in server_manager.pipeline_manager.pipelines.values():
        for model_info in pipeline.get_ai_models_info():
            if model_info.config_name:
                loaded_model_names.add(model_info.config_name)
            loaded_model_file_names.add(model_info.name)

    catalog = []
    for model_config in load_available_ai_models():
        config_name = model_config.get("yaml_file_name")
        file_name = model_config.get("model_file_name") or config_name
        entry = {
            "config_name": config_name,
            "name": file_name,
            "identifier": model_config.get("model_identifier"),
            "version": model_config.get("model_version"),
            "categories": _normalize_string_list(model_config.get("model_category")) or [],
            "type": model_config.get("model_type", model_config.get("type")),
            "capabilities": _resolve_capabilities(model_config),
            "supported_scopes": _resolve_supported_scopes(model_config),
            "active": config_name in active_model_names,
            "loaded": config_name in loaded_model_names or file_name in loaded_model_file_names,
            "pinned": config_name in pinned_model_names,
        }
        catalog.append(entry)

    catalog.sort(key=lambda item: (item["active"] is False, item["config_name"]))
    return catalog


def get_loaded_models(server_manager):
    catalog = get_model_catalog(server_manager)
    return [entry for entry in catalog if entry["active"] or entry["loaded"]]


def filter_pipeline_models(pipeline, requested_model_names=None):
    ai_models = pipeline.get_ai_models_info()
    requested = set(_normalize_string_list(requested_model_names) or [])
    if not requested:
        return ai_models
    filtered = []
    for model_info in ai_models:
        if model_info.config_name in requested or model_info.name in requested:
            filtered.append(model_info)
    return filtered


async def resolve_request_models(server_manager, want, default_scope, load_policy):
    if load_policy not in VALID_LOAD_POLICIES:
        raise ValueError(f"Unsupported load_policy '{load_policy}'. Expected one of {sorted(VALID_LOAD_POLICIES)}")

    requested_model_names = resolve_want_model_names(server_manager, want, default_scope)
    if not requested_model_names:
        return [], False

    active_model_names = set(load_active_ai_models())
    missing_models = [name for name in requested_model_names if name not in active_model_names]
    if not missing_models:
        return requested_model_names, False

    if load_policy == "use_loaded":
        raise RuntimeError(f"Requested models are not active: {missing_models}")

    next_active = list(active_model_names)
    next_active.extend(missing_models)
    next_active = _dedupe(next_active)
    pinned_models = _dedupe(load_pinned_ai_models())
    await server_manager.reload_active_models(active_ai_models=next_active, pinned_ai_models=pinned_models)
    return requested_model_names, True


async def load_models(server_manager, model_names):
    requested = _validate_model_names(model_names, server_manager)
    active_model_names = _dedupe(load_active_ai_models() + requested)
    pinned_model_names = _dedupe(load_pinned_ai_models())
    await server_manager.reload_active_models(active_ai_models=active_model_names, pinned_ai_models=pinned_model_names)
    return get_loaded_models(server_manager)


async def unload_models(server_manager, model_names):
    requested = _validate_model_names(model_names, server_manager)
    pinned_model_names = set(load_pinned_ai_models())
    blocked = sorted(pinned_model_names.intersection(requested))
    if blocked:
        raise RuntimeError(f"Cannot unload pinned models: {blocked}")

    active_model_names = [name for name in load_active_ai_models() if name not in requested]
    await server_manager.reload_active_models(active_ai_models=_dedupe(active_model_names), pinned_ai_models=sorted(pinned_model_names))
    return get_loaded_models(server_manager)


async def set_pinned_models(server_manager, model_names, pinned):
    requested = _validate_model_names(model_names, server_manager)
    config = dict(load_active_ai_config())
    active_model_names = _dedupe(config.get("active_ai_models", []) or [])
    pinned_model_names = set(config.get("pinned_ai_models", []) or [])

    if pinned:
        changed_active = False
        for model_name in requested:
            pinned_model_names.add(model_name)
            if model_name not in active_model_names:
                active_model_names.append(model_name)
                changed_active = True
        if changed_active:
            await server_manager.reload_active_models(active_ai_models=active_model_names, pinned_ai_models=sorted(pinned_model_names))
        else:
            config["pinned_ai_models"] = sorted(pinned_model_names)
            save_active_ai_config(config)
    else:
        pinned_model_names.difference_update(requested)
        config["pinned_ai_models"] = sorted(pinned_model_names)
        save_active_ai_config(config)

    return get_loaded_models(server_manager)


def resolve_want_model_names(server_manager, want, default_scope):
    want = want or []
    if not want:
        return []

    catalog = get_model_catalog(server_manager)
    catalog_by_name = {entry["config_name"]: entry for entry in catalog}
    requested_model_names = []

    for item in want:
        explicit_models = _normalize_string_list(getattr(item, "models", None)) or []
        if explicit_models:
            for model_name in explicit_models:
                if model_name not in catalog_by_name:
                    raise ValueError(f"Unknown model '{model_name}'")
            requested_model_names.extend(explicit_models)

        requested_capabilities = set(_normalize_string_list(getattr(item, "capability", None)) or [])
        requested_capabilities.update(_normalize_string_list(getattr(item, "capabilities", None)) or [])
        requested_scopes = set(_normalize_string_list(getattr(item, "scope", None)) or [])
        requested_scopes.update(_normalize_string_list(getattr(item, "scopes", None)) or [])
        if not requested_scopes and default_scope:
            requested_scopes.add(default_scope)

        matched = []
        if requested_capabilities or requested_scopes:
            for entry in catalog:
                if requested_capabilities and requested_capabilities.isdisjoint(entry["capabilities"]):
                    continue
                if requested_scopes and requested_scopes.isdisjoint(entry["supported_scopes"]):
                    continue
                matched.append(entry["config_name"])
            if not matched and not explicit_models:
                raise ValueError(
                    f"No models matched want item capabilities={sorted(requested_capabilities)} scopes={sorted(requested_scopes)}"
                )
            requested_model_names.extend(matched)

    return _dedupe(requested_model_names)


def _validate_model_names(model_names, server_manager):
    normalized = _normalize_string_list(model_names) or []
    catalog_names = {entry["config_name"] for entry in get_model_catalog(server_manager)}
    unknown = [name for name in normalized if name not in catalog_names]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")
    return normalized


def _normalize_string_list(raw_value):
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


def _resolve_capabilities(model_config):
    capabilities = _normalize_string_list(model_config.get("model_capabilities", model_config.get("capabilities")))
    if capabilities:
        return capabilities
    model_type = str(model_config.get("model_type", "")).lower()
    if "embed" in model_type:
        return ["embedding"]
    if "detect" in model_type:
        return ["detection"]
    if "classif" in model_type:
        return ["classification"]
    return ["tagging"]


def _resolve_supported_scopes(model_config):
    scopes = _normalize_string_list(model_config.get("supported_target_scopes", model_config.get("target_scopes")))
    return scopes or ["asset", "frame", "region"]


def _dedupe(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped