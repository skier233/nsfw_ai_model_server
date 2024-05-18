import yaml
import os

def load_config(file_path, default_config={}):
    """
    Load YAML configuration from the specified file path.
    Returns a dictionary with configuration values.
    """

    # Check if the file exists
    if not os.path.exists(file_path) and default_config is None:
        print(f"ERROR: Configuration file {file_path} not found. Please ensure this config file is present!!")
        return None
    elif not os.path.exists(file_path):
        print(f"WARNING: Configuration file {file_path} not found. Using default values.")
        return default_config

    try:
        # Open the YAML file and load it
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update the default configuration with values from the file
        # This ensures that any missing keys in the YAML file will use the defaults
        updated_config = {**default_config, **(config or {})}
        return updated_config
    except yaml.YAMLError as e:
        print(f"Error loading the YAML file: {e}")
        return default_config
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return default_config