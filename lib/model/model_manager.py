
from lib.async_lib.async_processing import ModelProcessor
from lib.config.config_utils import load_config
from lib.model.ai_model import AIModel
from lib.model.python_model import PythonModel
from lib.model.video_preprocessor import VideoPreprocessorModel
from lib.model.image_preprocessor import ImagePreprocessorModel


class ModelManager:
    def __init__(self):
        self.models = {}

    def get_or_create_model(self, modelName):
        if modelName not in self.models:
            self.models[modelName] = self.create_model(modelName)
        return self.models[modelName]
    
    def create_model(self, modelName):
        if not isinstance(modelName, str):
            raise ValueError("Model names must be strings that are the name of the model config file!")
        model_config_path = f"./config/models/{modelName}.yaml"
        try:
            model = model_factory(load_config(model_config_path))
        except Exception as e:
            print(f"Error loading model {model_config_path}: {e}")
            return None
        return model
    
def model_factory(model_config):
    match model_config["type"]:
        case "video_preprocessor":
            return ModelProcessor(VideoPreprocessorModel(model_config))
        case "image_preprocessor":
            return ModelProcessor(ImagePreprocessorModel(model_config))
        case "model":
            return ModelProcessor(AIModel(model_config))
        case "python":
            return ModelProcessor(PythonModel(model_config))
        case _:
            raise ValueError(f"Model type {model_config['type']} not recognized!")