from lib.async_lib.async_processing import QueueItem

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

def validate_string_list(input_list):
    if not isinstance(input_list, list):
        return False
    for item in input_list:
        if not isinstance(item, str):
            return False
    if len(input_list) == 0:
        return False
    return True
    