import time
import numpy as np
import torch

from lib.model.ai_model import AIModel


class AIAudioClassifierModel(AIModel):
    """AI model subclass for audio classification (AST via torch.export).

    Produces AudioSet-style classification scores for gating and type-binning.
    Output: [{"scores": {category: float, ...}, "top5": [(label, score), ...],
              "dominant_type": str, "classifier": "model_name"}]

    Input: mel-spectrogram tensor [1, n_mels, target_length].
    """

    def __init__(self, configValues):
        super().__init__(configValues, keep_on_device=False)
        self.fill_to_batch = configValues.get("fill_to_batch_size", False)
        self.num_classes = configValues.get("num_classes", 527)
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
                    raise ValueError("Audio classifier model expects tensor input")
                tensors.append(tensor)
                valid_items.append(item)
            except Exception as e:
                self.logger.error(f"Error preparing input for AIAudioClassifierModel: {e}")
                self.logger.debug("Stack trace:", exc_info=True)
                item_future.set_exception(e)

        if not tensors:
            return

        try:
            # Mel-spectrograms from preprocessor are [1, n_mels, T] — squeeze leading dim
            squeezed = []
            for t in tensors:
                if t.dim() == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)  # [n_mels, T]
                squeezed.append(t)

            # Pad to same time-frame length for batching
            max_t = max(t.shape[-1] for t in squeezed)
            padded = []
            for t in squeezed:
                if t.shape[-1] < max_t:
                    pad_size = max_t - t.shape[-1]
                    t = torch.nn.functional.pad(t, (0, pad_size))
                padded.append(t)

            batch = torch.stack(padded, dim=0)  # [B, n_mels, T]
            batch = batch.to(self.device)
            if batch.dtype != torch.float16:
                batch = batch.half()

            with torch.no_grad():
                logits = self.model.run_raw(batch)

            if isinstance(logits, torch.Tensor):
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

            for i, item in enumerate(valid_items):
                p = probs[i]
                top_idx = p.argsort()[-5:][::-1]
                top5 = [(int(idx), round(float(p[idx]), 4)) for idx in top_idx]

                result = [{
                    "probabilities": p.tolist(),
                    "top5": top5,
                    "num_classes": self.num_classes,
                    "classifier": self.model_file_name,
                }]
                await item.item_future.set_data(item.output_names[0], result)

        except Exception as e:
            self.logger.error(f"Error in AIAudioClassifierModel batch inference: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            for item in valid_items:
                item.item_future.set_exception(e)

        self.logger.debug(
            f"Classified {len(valid_items)} audio segments in {time.time() - batch_started_at:.3f}s "
            f"with {self.model_file_name}"
        )
