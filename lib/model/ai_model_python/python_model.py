import torch
from lib.utils.memory_utils import clear_gpu_cache

class PythonModel:
    def __init__(self, path, batch_size, device, fill_batch_size):
        self.model_path = path
        self.max_batch_size = batch_size
        self.fill_batch_size = fill_batch_size
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.xpu.is_available(): 
            self.device = torch.device('xpu')
        elif torch.backed.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self._model_loaded = False
        self.load_model()

        self.model_loaded = True
        
    def run_model(self, preprocessed_images, applySigmoid):
        preprocessed_images = preprocessed_images.to(self.device)
        original_batch_size = preprocessed_images.size(0)
        if self.fill_batch_size:
            if preprocessed_images.size(0) < self.max_batch_size:
                padding_size = self.max_batch_size - original_batch_size
                padding = torch.zeros((padding_size, *preprocessed_images.shape[1:]), device=self.device)
                preprocessed_images = torch.cat([preprocessed_images, padding], dim=0)
        if self.device.type in ['cuda', 'xpu'] and preprocessed_images.dtype != torch.float16:
            preprocessed_images = preprocessed_images.half()  # Convert to half precision
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.autocast("cuda", enabled=True):
                    output = self.model(preprocessed_images)
            elif self.device.type == 'xpu':
                with torch.autocast("xpu", enabled=True):
                    output = self.model(preprocessed_images)
            else:
                output = self.model(preprocessed_images)
            if applySigmoid:
                output = torch.sigmoid(output)
    
        # Remove the outputs corresponding to the padding images
        output = output[:original_batch_size]
        return output.cpu()

    def process_images(self, preprocessed_images, applySigmoid = True):
        if preprocessed_images.size(0) <= self.max_batch_size:
            return self.run_model(preprocessed_images, applySigmoid)
        else:
            chunks = torch.split(preprocessed_images, self.max_batch_size)
            results = []
            for chunk in chunks:
                chunk_result = self.run_model(chunk, applySigmoid)
                results.append(chunk_result)
            return torch.cat(results, dim=0)

    def load_model(self):
        if self.model_loaded:
            return
        self.model = torch.jit.load(self.model_path, map_location=self.device).to(self.device)
        self.model_loaded = True

    def unload_model(self):
        self.model = None
        clear_gpu_cache()
        self.model_loaded = False

    @property
    def model_loaded(self):
        return self._model_loaded
    
    @model_loaded.setter
    def model_loaded(self, value):
        if isinstance(value, bool):
            self._model_loaded = value
        else:
            raise ValueError("model_loaded must be a boolean value")
        
    def __enter__(self):
        """
        Context management method to use with 'with' statements.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context management method to close the TensorBoard writer upon exiting the 'with' block.
        """
        del self.model
        clear_gpu_cache()