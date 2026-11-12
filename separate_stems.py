"""Separate songs into stems with a pretrained Demucs model and cache them on disk.
 
Objective 1, PI1 (basic): load and preprocess audio, separate each song into
vocals / drums / bass / guitar / piano / other with a pretrained Demucs model
(no training), and cache the stems.
 
Reads the ``manifest.csv`` written by ``harmonix_download.py`` and processes
every row whose status is ``ok``.
 
Cache layout::
 
    <stems_root>/<dataset>/<file_id>/vocals.flac
                                    drums.flac
                                    bass.flac
                                    guitar.flac
                                    piano.flac
                                    other.flac
                                    metadata.json
 
"""

import argparse
import csv
import shutil
import time
from pathlib import Path
 
import pandas as pd
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model

LOG_COLUMNS = ["file_id", "status", "time_s", "residual", "error"]


class StemSeparator:
    """Wrapper for a pretrained Demucs model (default is the 6-stem model). Takes an audio file as input, returns named stems."""

    def __init__(self, model_name:str = "htdemucs_6s"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(model_name).to(self.device).eval()
        self.sample_rate = self.model.samplerate

    def separate(self, path: str):
        """Takes one audio file and returns a dictionary containing its stems, plus a faithfullness number."""
        # load and convert audio to model specs (sr = 44.1 kHz, stereo audio)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mix = convert_audio(torch.from_numpy(data.T), sr, self.sample_rate, self.model.audio_channels)

        # standardize audio (ensure it has a consistent scale). Demucs performs more reliably on audio with a consistent scale.
        ref = mix.mean(0)
        normalized = (mix - ref.mean()) / ref.std()

        # run model
        with torch.inference_mode():
            sources = apply_model(self.model, normalized[None], device=self.device, split=True, progress=False)[0] # [0] drops the batch dimension

        # undo standardization (apply scaling in reverse)
        sources = sources.cpu() * ref.std() + ref.mean()

        # measure reconstruction error (faithfullness)
        # if separation is perfect, adding the stems together results in the original mix (residual = 0). 
        residual = ((sources.sum(0) - mix).norm() / mix.norm()).item()

        # TODO: if the stem is mostly silent, discard it

        return dict(zip(self.model.sources, self._prevent_clipping(sources))), residual

    @staticmethod
    def _prevent_clipping(sources: torch.Tensor) -> torch.Tensor:
        """Scales everything down slightly if any sample exceeds the maximum level (1.0) to prevent distortions."""
        peak = sources.abs().max()
        return sources / (1.01 * peak) if peak > 1 else sources


class StemCache:
    """Stores stems as <root>/<file_id>/<stem>.flac. A folder exists only if fully written."""
 
    def __init__(self, root: Path):
        self.root = root
 
    def has(self, file_id: str) -> bool:
        return (self.root / file_id).is_dir()
 
    def save(self, file_id: str, stems: dict, sample_rate: int):
        tmp = self.root / f"{file_id}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)   # leftover from an interrupted run
        tmp.mkdir(parents=True)

        for name, audio in stems.items():
            sf.write(tmp / f"{name}.flac", audio.T.numpy(), sample_rate, subtype="PCM_16")
        tmp.rename(self.root / file_id)

class RunLog:
    """Appends one CSV row per processed track."""
 
    def __init__(self, path: Path):
        self.path = path
 
    def add(self, **row):
        is_new = not self.path.exists()

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, LOG_COLUMNS, restval="")

            if is_new:
                writer.writeheader()

            writer.writerow(row)

class SeparationPipeline:
    """Pipeline for conducting audio separation.
    1. Calls StemSeparator to turn audio into stems
    2. Calls StemCache to store stems on disk
    3. Calls RunLog to record data"""

    def __init__(self, separator: StemSeparator, cache: StemCache, log: RunLog):
        self.separator = separator
        self.cache = cache
        self.log = log
 
    def run(self, tracks: list):
        """Main loop. Takes tracks: list of (file_id, audio_path)."""
        
        for i, (file_id, path) in enumerate(tracks, 1):
            if self.cache.has(file_id):
                # audio has already been processed, skip
                print(f"[{i}/{len(tracks)}] skip {file_id} (cached)")
                continue

            print(f"[{i}/{len(tracks)}] {file_id}")
            self._process(file_id, path)
 
    def _process(self, file_id: str, path: str):
        start = time.perf_counter()

        try:
            # get stems from this track, save to cache
            stems, residual = self.separator.separate(path)
            self.cache.save(file_id, stems, self.separator.sample_rate)

        except Exception as e:
            error = str(e).replace("\n", " ")[:200]
            print(f"  FAILED {file_id}: {error}")
            self.log.add(file_id=file_id, status="failed", error=error)
            return

        self.log.add(file_id=file_id, status="ok",
                     time_s=round(time.perf_counter() - start, 2),
                     residual=round(residual, 4))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="harmonix_audio/manifest.csv")
    p.add_argument("--out", default="stems/harmonix", help="stem cache directory")
    p.add_argument("--model", default="htdemucs_6s")
    p.add_argument("--limit", type=int, default=None, help="only first N tracks (for testing)")
    args = p.parse_args()
 
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["status"] == "ok"].head(args.limit)
    tracks = list(zip(manifest["file_id"], manifest["path"]))
 
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
 
    pipeline = SeparationPipeline(StemSeparator(args.model),
                                  StemCache(out),
                                  RunLog(out / "separation_log.csv"))
    pipeline.run(tracks)
    print("Done.")
 
 
if __name__ == "__main__":
    main()
  
