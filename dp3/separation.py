
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model

from . import config

class StemSeparator:
    """Wrapper for a pretrained Demucs model (default is the 6-stem model). Takes an audio file as input, returns named stems."""

    def __init__(self, model_name:str = config.SEPARATION_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(model_name).to(self.device).eval()
        self.sample_rate = self.model.samplerate

    def separate(self, path: str) -> tuple:
        """Separates one audio file into stems. Returns (stems, residual)."""

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return self.separate_array(data.T, sr)

    def separate_array(self, audio: np.ndarray, sr: int) -> tuple:
        """Separate in-memory audio shaped (channels, samples) at sample rate sr.
 
        Returns (stems, residual): stems maps name -> tensor (channels, samples)
        at self.sample_rate, and residual is the relative reconstruction error
        (0 if the stems sum exactly to the mix).
        """

        # Convert audio to model specs (sr = 44.1 kHz, stereo audio)
        mix = convert_audio(torch.from_numpy(audio), sr, self.sample_rate, self.model.audio_channels)

        # Standardize audio (ensure it has a consistent scale). Demucs performs more reliably on audio with a consistent scale.
        ref = mix.mean(0)
        normalized = (mix - ref.mean()) / ref.std()

        # Run model
        with torch.inference_mode():
            sources = apply_model(self.model, normalized[None], device=self.device, split=True, progress=False)[0] # [0] drops the batch dimension

        # Undo standardization (apply scaling in reverse)
        sources = sources.cpu() * ref.std() + ref.mean()

        # Measure reconstruction error (faithfullness)
        # If separation is perfect, adding the stems together results in the original mix (residual = 0). 
        residual = ((sources.sum(0) - mix).norm() / mix.norm()).item()


        return dict(zip(self.model.sources, self._prevent_clipping(sources))), residual

    @staticmethod
    def _prevent_clipping(sources: torch.Tensor) -> torch.Tensor:
        """Scales everything down slightly if any sample exceeds the maximum level (1.0) to prevent distortions."""

        peak = sources.abs().max()
        return sources / (1.01 * peak) if peak > 1 else sources



