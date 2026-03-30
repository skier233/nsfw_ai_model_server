import copy
import os
import yaml


model_capabilities_yaml_path = "./config/model_capabilities.yaml"


DEFAULT_MODEL_CAPABILITIES_CONFIG = {
    "image_pipeline_dynamic_v3": {
        "full_image_models": "ALL",
        "detector_models": [],
        "region_models": {},
    },
    "video_pipeline_dynamic_v3": {
        "full_image_models": "ALL",
        "detector_models": [],
        "region_models": {},
    },
    "image_pipeline_face_embeddings_v1": {
        "full_image_models": [],
        "detector_models": ["face_detector_torchexport"],
        "region_models": {
            "face_detector_torchexport": ["face_embedding_torchexport"],
        },
    },
    "video_pipeline_face_recognition_v1": {
        "full_image_models": [],
        "detector_models": ["face_detector_torchexport"],
        "region_models": {
            "face_detector_torchexport": ["face_embedding_torchexport"],
        },
    },
}


def ensure_model_capabilities_config():
    if os.path.exists(model_capabilities_yaml_path):
        return

    os.makedirs(os.path.dirname(model_capabilities_yaml_path), exist_ok=True)
    with open(model_capabilities_yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(DEFAULT_MODEL_CAPABILITIES_CONFIG, handle, default_flow_style=False, sort_keys=False)


def load_model_capabilities_config():
    ensure_model_capabilities_config()
    with open(model_capabilities_yaml_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or copy.deepcopy(DEFAULT_MODEL_CAPABILITIES_CONFIG)


def save_model_capabilities_config(config):
    os.makedirs(os.path.dirname(model_capabilities_yaml_path), exist_ok=True)
    with open(model_capabilities_yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
