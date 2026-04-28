"""AudioPreprocessSpec — descriptor for audio preprocessing in the pipeline.

Analogous to PreprocessSpec for images.  Each audio AI model declares what
preprocessed tensor it needs (sample rate, duration, mel-spectrogram config,
device, precision).  The audio preprocessor collects all unique specs and
produces one tensor per spec from a single ffmpeg extraction.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AudioPreprocessSpec:
    """Immutable descriptor for one preprocessed audio tensor."""

    sample_rate: int = 16000
    duration_seconds: float = 4.0       # Window duration in seconds.
    device: str = "gpu"
    half_precision: bool = False

    # Fbank mode (ECAPA-TDNN, etc.) — SpeechBrain-style Fbank features.
    use_fbank: bool = False
    n_fbank: int = 80                   # Number of filterbank channels.

    # Mel-spectrogram mode (AST, BEATs, etc.)
    use_mel_spectrogram: bool = False
    n_mels: int = 128
    target_length: int = 1024           # Number of time frames in spectrogram.

    @property
    def key(self) -> str:
        """Deterministic, human-readable pipeline-data key."""
        prec = "f16" if self.half_precision else "f32"
        if self.use_fbank:
            mode = f"fbank{self.n_fbank}"
        elif self.use_mel_spectrogram:
            mode = f"mel{self.n_mels}x{self.target_length}"
        else:
            mode = "wav"
        return f"audio_prep__{self.sample_rate}hz_{self.duration_seconds}s_{mode}_{self.device}_{prec}"

    @staticmethod
    def for_model(model_inner) -> "AudioPreprocessSpec":
        """Derive a spec from a model's preprocess_config."""
        pc = getattr(model_inner, "preprocess_config", None) or {}
        return AudioPreprocessSpec(
            sample_rate=pc.get("sample_rate", 16000),
            duration_seconds=pc.get("duration_seconds", 4.0),
            device=pc.get("device", "gpu"),
            half_precision=pc.get("half_precision", False),
            use_fbank=pc.get("use_fbank", False),
            n_fbank=pc.get("n_fbank", 80),
            use_mel_spectrogram=pc.get("use_mel_spectrogram", False),
            n_mels=pc.get("n_mels", 128),
            target_length=pc.get("target_length", 1024),
        )
