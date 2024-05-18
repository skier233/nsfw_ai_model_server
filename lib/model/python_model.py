from lib.model.model import Model
from importlib import import_module

class PythonModel(Model):
    def __init__(self, configValues):
        Model.__init__(self, configValues)
        self.function_name = configValues.get("function_name", None)
        if self.function_name is None:
            raise ValueError("function_name is required for models of type python")
        module_name = "lib.model.python_functions"
        try:
            module = import_module(module_name)
            self.function = getattr(module, self.function_name)
        except ImportError:
            raise ImportError(f"Module '{module_name}' not found.")
        except AttributeError:
            raise AttributeError(f"Function '{self.function_name}' not found in module '{module_name}'.")

    async def worker_function(self, data):
        await self.function(data)