import time
import numpy as np
import torch

from lib.model.ai_model import AIModel


class AIVisualEmbeddingModel(AIModel):
    """AI model subclass for visual embedding (DINOv3, MetaCLIP2, etc. via torch.export).

    Produces a dense embedding vector for each input image/frame.
    Output: [{"vector": [...], "norm": float, "embedder": "model_name", "dim": int}]

    Works on full images (not face regions). Supports batched inference.
    """

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)
        self.fill_to_batch = configValues.get("fill_to_batch_size", False)
        self.embedding_dim = configValues.get("embedding_dim", None)
        self.l2_normalize = configValues.get("l2_normalize", True)
        self.preprocess_config = configValues.get("preprocess_config", None)

    async def worker_function(self, data):
        batch_started_at = time.time()

        # Collect tensors for batched inference
        tensors = []
        valid_items = []
        for item in data:
            item_future = item.item_future
            try:
                tensor = item_future[item.input_names[0]]
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("Visual embedding model expects tensor input")
                tensors.append(tensor)
                valid_items.append(item)
            except Exception as e:
                self.logger.error(f"Error preparing input for AIVisualEmbeddingModel: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                item_future.set_exception(e)

        if not tensors:
            return

        try:
            batch = torch.stack(tensors, dim=0).to(self.device)
            if batch.dtype != torch.float16:
                batch = batch.half()
            with torch.no_grad():
                embeddings = self.model.run_raw(batch)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            if self.l2_normalize:
                norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                embeddings = embeddings / norms

            for i, item in enumerate(valid_items):
                embedding = embeddings[i]
                norm = float(np.linalg.norm(embedding))
                dim = int(embedding.shape[-1])
                result = [{
                    "vector": embedding.tolist(),
                    "norm": norm,
                    "dim": dim,
                    "embedder": self.model_file_name,
                }]
                await item.item_future.set_data(item.output_names[0], result)

        except Exception as e:
            self.logger.error(f"Error in AIVisualEmbeddingModel batch inference: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            for item in valid_items:
                item.item_future.set_exception(e)

        self.logger.debug(
            f"Embedded {len(valid_items)} images in {time.time() - batch_started_at:.3f}s "
            f"with {self.model_file_name}"
        )
