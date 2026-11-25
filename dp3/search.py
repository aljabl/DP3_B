"""Search indexed stems with the descriptors from :mod:`dp3.features`."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .features import FeatureExtractor, to_analysis
from .stem_service import DATA_ROOT
from .upload import CLIP_SECONDS, CLIP_TOLERANCE_SECONDS, UploadedAudio, source_matches_file
from .facets import FacetPolicy


class SearchError(ValueError):
    """The query or saved search index cannot produce useful matches."""


@dataclass(frozen=True)
class SearchResult:
    dataset: str
    track_id: str
    stem: str
    similarity: float
    mix_path: Path
    stem_path: Path

    @property
    def title(self) -> str:
        return self.mix_path.stem if self.dataset != "fma" else self.track_id.partition("_")[2]


def _read_audio(data: bytes) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    return samples.T, sample_rate


def query_descriptors(mix_bytes: bytes, stem_bytes: bytes, stem: str) -> dict[str, np.ndarray]:
    """Use the offline pipeline's beat grid, chroma, and MFCC extraction."""
    mix, mix_rate = _read_audio(mix_bytes)
    isolated, stem_rate = _read_audio(stem_bytes)
    analysis = FeatureExtractor().analyze(
        to_analysis(mix, mix_rate),
        {stem: to_analysis(isolated, stem_rate)},
    )

    result = analysis["stems"][stem]
    if result["status"] != "ok":
        raise SearchError(f"The {stem} stem is too quiet to search. Choose another stem.")

    facets = FacetPolicy().facets_for(stem)
    return {facet: result[facet] for facet in facets}

class SimilarityIndex:
    """The saved raw embeddings and their playable audio files."""

    def __init__(self, rows: pd.DataFrame, vectors: dict[str, np.ndarray], data_root: Path,
                 policy: FacetPolicy | None = None):
        self.rows = rows
        self.vectors = vectors
        self.data_root = data_root
        self.policy = policy or FacetPolicy()
        self.source_paths = {}
        for dataset in ("previews", "harmonix"):
            for path in (data_root / "audio" / dataset).glob("*.wav"):
                self.source_paths[(dataset, path.stem.replace(" ", ""))] = path

    @classmethod
    def load(cls, data_root: Path = DATA_ROOT) -> "SimilarityIndex":
        policy = FacetPolicy()
        index_dir = data_root / "index"

        try:
            rows = pd.read_csv(index_dir / "embeddings_index.csv", dtype={"track_id": str})
            with np.load(index_dir / "embeddings.npz", allow_pickle=False) as saved:
                vectors = {facet: saved[facet] for facet in policy.indexed_facets()}

        except (FileNotFoundError, KeyError) as exc:
            raise SearchError("The stem search index is missing. Run the table build stage first.") from exc

        return cls(rows, vectors, data_root, policy)

    def _mix_path(self, dataset: str, track_id: str) -> Path | None:
        if dataset == "fma":
            number = track_id.partition("_")[0]
            if len(number) != 6 or not number.isdigit():
                return None
            return self.data_root / "audio" / "fma" / "fma_small" / number[:3] / f"{number}.mp3"
        return self.source_paths.get((dataset, track_id))

    def search(
        self,
        stem: str,
        query: dict[str, np.ndarray],
        *,
        upload: UploadedAudio | None = None,
        limit: int = 3,
    ) -> list[SearchResult]:
        """Return the most similar playable stems of the same type."""

        try:
            facets = self.policy.facets_for(stem)

        except ValueError as exc:
            raise SearchError(str(exc)) from exc

        if not facets:
            raise SearchError(f"No search facets are available for {stem} yet.")

        missing = [facet for facet in facets if facet not in query]
        if missing:
            raise SearchError(f"The query is missing: {', '.join(missing)}.")

        candidates = self.rows[(self.rows["stem"] == stem) & (self.rows["status"] == "ok")]
        if candidates.empty:
            raise SearchError(f"No indexed {stem} stems are available.")

        positions = candidates["row"].to_numpy(dtype=int)
        if (positions < 0).any() or (positions >= len(self.vectors[facets[0]])).any():
            raise SearchError("The saved search index has invalid row numbers.")

        candidate_parts = []
        query_parts = []
        for facet in facets:
            values = self.vectors[facet][positions]
            requested = np.asarray(query[facet], dtype=np.float32)
            if requested.shape != (values.shape[1],):
                raise SearchError(f"The {facet} query does not match the saved index.")

            mean = values.mean(axis=0)
            scale = values.std(axis=0) + 1e-8
            weight = np.sqrt(values.shape[1])
            candidate_parts.append((values - mean) / scale / weight)
            query_parts.append((requested - mean) / scale / weight)

        matrix = np.hstack(candidate_parts)
        query_vector = np.concatenate(query_parts)
        query_norm = np.linalg.norm(query_vector)

        if not np.isfinite(query_norm) or query_norm < 1e-12:
            raise SearchError("The selected stem has no usable search features.")

        norms = np.linalg.norm(matrix, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = (matrix @ query_vector) / (norms * query_norm)

        scores = np.where(np.isfinite(scores), scores, -np.inf)

        results = []
        for position in np.argsort(-scores, kind="stable"):
            if len(results) == limit or not np.isfinite(scores[position]):
                break
            row = candidates.iloc[int(position)]
            dataset, track_id = str(row["dataset"]), str(row["track_id"])
            mix_path = self._mix_path(dataset, track_id)
            stem_path = self.data_root / "stems" / dataset / track_id / f"{stem}.flac"
            if mix_path is None or not mix_path.is_file() or not stem_path.is_file():
                continue
            if (
                upload is not None
                and upload.clip_start_seconds == 0
                and abs(upload.duration_seconds - CLIP_SECONDS) <= CLIP_TOLERANCE_SECONDS
                and source_matches_file(mix_path, upload)
            ):
                continue
            results.append(SearchResult(dataset, track_id, stem, float(scores[position]), mix_path, stem_path))
        return results
