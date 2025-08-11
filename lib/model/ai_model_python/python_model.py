import gc
import torch


class PythonModel:
    """
    TorchScript model runner used by AIModel.

    Constructor matches AIModel.load():
        PythonModel(path, batch_size, device, fill_batch_size)

    Efficiency features:
    - Device auto-select (honors provided device).
    - Non-blocking H2D copies when possible.
    - Optional batch fill to max_batch_size.
    - FP16 + autocast on CUDA.
    - cuDNN benchmark for fixed-size convs.
    """

    def __init__(self, path, batch_size, device, fill_batch_size):
        self.model_path = path
        self.max_batch_size = int(batch_size) if batch_size else 1
        self.fill_batch_size = bool(fill_batch_size)

        # cuDNN autotune for static shapes
        torch.backends.cudnn.benchmark = True

        # Device (no MPS per your request)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            try:
                name = torch.cuda.get_device_name(self.device)
            except Exception:
                name = "Unknown CUDA device"
            print(f"[PythonModel] Using device: {self.device} ({name})")
        else:
            print(f"[PythonModel] Using device: {self.device}")

        self._model_loaded = False
        self.load_model()
        self.model_loaded = True

    def _log_mem(self, tag):
        if self.device.type == "cuda":
            try:
                alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                reserv = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                print(f"[PythonModel] {tag} | allocated={alloc:.1f}MB reserved={reserv:.1f}MB")
            except Exception:
                pass

    def run_model(self, preprocessed_images, applySigmoid: bool):
        # Inputs to target device (non_blocking helps if source is pinned)
        preprocessed_images = preprocessed_images.to(self.device, non_blocking=True)
        original_batch_size = preprocessed_images.size(0)

        # Optional pad to full batch size
        if self.fill_batch_size and original_batch_size < self.max_batch_size:
            padding_size = self.max_batch_size - original_batch_size
            padding = torch.zeros(
                (padding_size, *preprocessed_images.shape[1:]),
                device=self.device,
                dtype=preprocessed_images.dtype,
            )
            preprocessed_images = torch.cat([preprocessed_images, padding], dim=0)

        # Prefer FP16 on CUDA
        if self.device.type == "cuda" and preprocessed_images.dtype != torch.float16:
            preprocessed_images = preprocessed_images.half()

        self._log_mem("before forward")

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda", enabled=True):
                    output = self.model(preprocessed_images)
            else:
                output = self.model(preprocessed_images)

            if applySigmoid:
                output = torch.sigmoid(output)

        # Remove padding rows
        output = output[:original_batch_size]

        self._log_mem("after forward")

        # Return on CPU for compatibility with existing pipeline
        return output.cpu()

    def process_images(self, preprocessed_images, applySigmoid: bool = True):
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
        # TorchScript load
        self.model = torch.jit.load(self.model_path, map_location=self.device).to(self.device)
        self.model.eval()
        try:
            p = next(self.model.parameters())
            print(f"[PythonModel] Model loaded on: {p.device}")
        except Exception:
            print(f"[PythonModel] Model loaded (TorchScript) on: {self.device}")
        self.model_loaded = True

    def unload_model(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            del self.model
        except AttributeError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
