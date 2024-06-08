from lib.async_lib.async_processing import QueueItem
from lib.model.ai_model import AIModel
from lib.model.video_preprocessor import VideoPreprocessorModel

class ModelWrapper:
    def __init__(self, model, inputs, outputs):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs

class Pipeline:
    def __init__(self, configValues, model_manager):
        if not validate_string_list(configValues["inputs"]):
            raise ValueError("Error: Pipeline inputs must be a non-empty list of strings!")
        if not configValues["output"]:
            raise ValueError("Error: Pipeline output must be a non-empty string!")
        if not isinstance(configValues["models"], list):
            raise ValueError("Error: Pipeline models must be a non-empty list of strings!")
        
        self.short_name = configValues.get("short_name", None)
        if self.short_name is None:
            raise ValueError("Error: Pipeline short_name must be a non-empty string!")
        self.version = configValues.get("version", None)
        if self.version is None:
            raise ValueError("Error: Pipeline version must be a non-empty float!")
        self.inputs = configValues["inputs"]
        self.output = configValues["output"]

        self.models = []
        for model in configValues["models"]:
            if not validate_string_list(model["inputs"]):
                raise ValueError("Error: Model inputs must be a non-empty list of strings!")
            if not model["name"]:
                raise ValueError("Error: Model name must be a non-empty string!")
            modelName = model["name"]
            returned_model = model_manager.get_or_create_model(modelName)
            self.models.append(ModelWrapper(returned_model, model["inputs"], model["outputs"]))
    
    async def event_handler(self, itemFuture, key):
        if key == self.output:
            itemFuture.close_future(itemFuture[key])
        for model in self.models:
            if key in model.inputs:
                allOtherInputsPresent = all(inputName in itemFuture.data for inputName in model.inputs if inputName != key)
                if allOtherInputsPresent:
                    await model.model.add_to_queue(QueueItem(itemFuture, model.inputs, model.outputs))

    async def start_model_processing(self):
        for model in self.models:
            await model.model.start_workers()

    def get_first_video_preprocessor(self):
        for model in self.models:
            if isinstance(model.model.model, VideoPreprocessorModel):
                return model.model.model
        return None
    
    def get_first_ai_model(self):
        for model in self.models:
            if isinstance(model.model.model, AIModel):
                return model.model.model
        return None

def validate_string_list(input_list):
    if not isinstance(input_list, list):
        return False
    for item in input_list:
        if not isinstance(item, str):
            return False
    if len(input_list) == 0:
        return False
    return True
    