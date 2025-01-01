import os
import yaml

model_directory = "./config/models"
model_fields_to_add = {
    "gentler_river": {"model_category": ["actions"], "model_version": 2.0, "model_image_size": 512, "model_info": "Most Accurate", "model_identifier": 200},
    "gentler_river_full": {"model_category": ["actions"], "model_version": 2.0, "model_image_size": 512, "model_info": "(CPU Variant if no Nvidia GPU available) Most Accurate", "model_identifier": 900},
    "vivid_galaxy": {"model_category": ["actions"], "model_version": 1.9, "model_image_size": 512, "model_info": "Free Variant", "model_identifier": 950, "normalization_config": 0},
    "vivid_galaxy_full": {"model_category": ["actions"], "model_version": 1.9, "model_image_size": 512, "model_info": "(CPU Variant if no Nvidia GPU available) Free Variant", "model_identifier": 970, "normalization_config": 0},
    "distinctive_haze": {"model_category": ["actions"], "model_version": 2.0, "model_image_size": 384, "model_info": "Faster but slightly less accurate", "model_identifier": 400},
    "iconic_sky": {"model_category": ["bodyparts"], "model_version": 0.5, "model_image_size": 512, "model_info": "Most Accurate", "model_identifier": 200},
    "true_lake": {"model_category": ["bdsm"], "model_version": 0.7, "model_image_size": 512, "model_info": "Most Accurate", "model_identifier": 200},
}

def migrate_to_2_0():
    for root, _, files in os.walk(model_directory):
        for file in files:
            if file.endswith('.yaml'):
                yaml_path = os.path.join(root, file)
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                if data.get('type') == 'model':
                    model_file_name = data.get('model_file_name')
                    if model_file_name in model_fields_to_add:
                        fields_to_add = model_fields_to_add[model_file_name]

                        updated = False
                        for key, value in fields_to_add.items():
                            if key not in data:
                                data[key] = value
                                updated = True

                        if updated:
                            with open(yaml_path, 'w') as f:
                                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    migrate_to_2_0()