

from lib.config.config_utils import load_config


post_processing_config = load_config("./config/post_processing/post_processing_config.yaml")

def get_or_default(dict, key, default):
    if key in dict and dict[key]:
        return dict[key]
    else:
        csv_defaults = post_processing_config.get("csv_defaults", {})
        return csv_defaults.get(key, default)