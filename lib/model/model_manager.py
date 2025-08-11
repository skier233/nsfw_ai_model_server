
import logging
from lib.async_lib.async_processing import ModelProcessor
from lib.config.config_utils import load_config
from lib.model.ai_model import AIModel
from lib.model.python_model import PythonModel
from lib.model.video_preprocessor import VideoPreprocessorModel
from lib.model.image_preprocessor import ImagePreprocessorModel


class ModelManager:
    def __init__(self):
        self.models = {}
        self.logger = logging.getLogger("logger")
        self.ai_models = []

    def get_or_create_model(self, modelName):
        if modelName not in self.models:
            self.models[modelName] = self.create_model(modelName)
        return self.models[modelName]
    
    def get_and_refresh_model(self, modelName):
        if modelName not in self.models:
            self.models[modelName] = self.create_model(modelName)
        else:
            del self.models[modelName]
            self.models[modelName] = self.create_model(modelName)
        return self.models[modelName]
    
    def create_model(self, modelName):
        if not isinstance(modelName, str):
            raise ValueError("Model names must be strings that are the name of the model config file!")
        model_config_path = f"./config/models/{modelName}.yaml"
        try:
            model = self.model_factory(load_config(model_config_path))
        except Exception as e:
            self.logger.error(f"Error loading model {model_config_path}: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            return None
        return model
    
    def model_factory(self, model_config):
        match model_config["type"]:
            case "video_preprocessor":
                return ModelProcessor(VideoPreprocessorModel(model_config))
            case "image_preprocessor":
                return ModelProcessor(ImagePreprocessorModel(model_config))
            case "model":
                model_processor = ModelProcessor(AIModel(model_config))
                self.ai_models.append(model_processor)
                model_count = len(self.ai_models)
                if model_count > 1:
                    for model_processor in self.ai_models:
                        ai_model = model_processor.model
                        ai_model.update_batch_with_mutli_models(model_count)
                        model_processor.update_values_from_child_model()


                return model_processor
            case "python":
                return ModelProcessor(PythonModel(model_config))
            case _:
                raise ValueError(f"Model type {model_config['type']} not recognized!")