import time
import numpy as np
import torch

from lib.model.ai_model import AIModel


class AIAudioEmbeddingModel(AIModel):
    """AI model subclass for audio/speaker embedding (ECAPA-TDNN via torch.export).

    Produces a dense embedding vector for each input audio segment.
    Output: [{"vector": [...], "norm": float, "embedder": "model_name", "dim": int}]

    Input: Fbank features tensor [1, num_frames, n_fbank] (pre-computed by
    the audio preprocessor).  The exported model takes [B, T, 80] and
    returns [B, 192].
    """

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)
        self.fill_to_batch = configValues.get("fill_to_batch_size", False)
        self.embedding_dim = configValues.get("embedding_dim", None)
        self.l2_normalize = configValues.get("l2_normalize", True)
        self.preprocess_config = configValues.get("preprocess_config", None)

    async def worker_function(self, data):
        batch_started_at = time.time()

        tensors = []
        valid_items = []
        for item in data:
            item_future = item.item_future
            try:
                tensor = item_future[item.input_names[0]]
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("Audio embedding model expects tensor input")
                tensors.append(tensor)
                valid_items.append(item)
            except Exception as e:
                self.logger.error(f"Error preparing input for AIAudioEmbeddingModel: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                item_future.set_exception(e)

        if not tensors:
            return

        try:
            # Fbank features are [1, T, 80] — pad along time dimension for batching
            max_len = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                if t.dim() == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)  # [T, 80]
                if t.shape[0] < max_len:
                    pad_size = max_len - t.shape[0]
                    t = torch.nn.functional.pad(t, (0, 0, 0, pad_size))
                padded.append(t)

            # Stack into batch [B, T, 80]
            batch = torch.stack(padded, dim=0)
            batch = batch.to(self.device)

            # ECAPA-TDNN is exported in float32 — bypass PythonModel.run_raw()
            # which would convert input to half and break the float32 graph.
            with torch.no_grad():
                embeddings = self.model.model(batch)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # L2 normalize
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
            self.logger.error(f"Error in AIAudioEmbeddingModel batch inference: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            for item in valid_items:
                item.item_future.set_exception(e)

        self.logger.debug(
            f"Embedded {len(valid_items)} audio segments in {time.time() - batch_started_at:.3f}s "
            f"with {self.model_file_name}"
        )
