"""Search indexed stems with the descriptors from :mod:`dp3.features`."""

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .features import FeatureExtractor, to_analysis
from .stem_service import DATA_ROOT
from .upload import (CLIP_SECONDS, CLIP_TOLERANCE_SECONDS, UploadedAudio,
                     bytes_match_file, song_identity, song_tags_from_media,
                     source_matches_file)
from .facets import FacetPolicy
from . import config
from .rhythm import RHYTHM_BLOCKS


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
    artist: str | None = None
    song_title: str | None = None

    @property
    def title(self) -> str:
        return self.song_title or (self.mix_path.stem if self.dataset != "fma"
                                   else self.track_id.partition("_")[2])

    @property
    def display_artist(self) -> str:
        return self.artist or "Unknown Artist"


def _read_audio(data: bytes) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    return samples.T, sample_rate


def query_descriptors(mix_bytes: bytes, stem_bytes: bytes, stem: str) -> dict[str, np.ndarray]:
    """Use the same facet extractors as the offline index build."""
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
                 policy: FacetPolicy | None = None,
                 user_source_paths: dict[str, Path] | None = None,
                 user_artists: dict[str, str | None] | None = None,
                 user_titles: dict[str, str] | None = None):
        self.rows = rows
        self.vectors = vectors
        self.data_root = data_root
        self.policy = policy or FacetPolicy()
        self.user_artists = user_artists or {}
        self.user_titles = user_titles or {}
        self.source_paths = {}
        for dataset in ("previews", "harmonix"):
            for path in (data_root / "audio" / dataset).glob("*.wav"):
                self.source_paths[(dataset, path.stem.replace(" ", ""))] = path
        for track_id, path in (user_source_paths or {}).items():
            self.source_paths[("user_added", track_id)] = path

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

        user_paths = {}
        user_artists = {}
        user_titles = {}
        additions = []
        user_features_dir = data_root / "features" / "user_added"
        indexed_ids = set(rows.loc[rows["dataset"] == "user_added", "track_id"].astype(str))
        for folder in sorted(user_features_dir.iterdir()) if user_features_dir.is_dir() else ():
            if not folder.is_dir() or folder.name in indexed_ids:
                continue
            try:
                meta = json.loads((folder / "meta.json").read_text())
                filename = meta["song_filename"]
                if Path(filename).name != filename:
                    continue
                song_path = data_root / "audio" / "user_added" / filename
                if not song_path.is_file():
                    continue
                with np.load(folder / "vectors.npz", allow_pickle=False) as saved:
                    extra = {facet: saved[facet] for facet in policy.indexed_facets()}
                count = len(next(iter(extra.values())))
                if any(array.ndim != 2 or len(array) != count or
                       array.shape[1] != vectors[facet].shape[1]
                       for facet, array in extra.items()):
                    continue
                offset = len(next(iter(vectors.values())))
                entry_rows = []
                for item in meta["rows"]:
                    local_row = int(item["row"])
                    if local_row < -1 or local_row >= count:
                        raise ValueError("Invalid row in saved upload features")
                    entry_rows.append({
                        "dataset": "user_added", "track_id": folder.name,
                        "stem": item["stem"], "status": item["status"],
                        "rms_db": item["rms_db"],
                        "row": offset + local_row if local_row >= 0 else -1,
                        "meter_beats": meta.get("meter_beats"),
                        "meter_confidence": meta.get("meter_confidence", 0.0),
                    })
                if not all((data_root / "stems" / "user_added" / folder.name /
                            f"{item['stem']}.flac").is_file() for item in meta["rows"]):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            additions.extend(entry_rows)
            for facet in policy.indexed_facets():
                vectors[facet] = np.concatenate((vectors[facet], extra[facet]))
            user_paths[folder.name] = song_path
            tagged_title, tagged_artist = ((None, None) if "title" in meta and "artist" in meta
                                           else song_tags_from_media(song_path))
            user_titles[folder.name], user_artists[folder.name] = song_identity(
                song_path.name, meta.get("title") or tagged_title,
                meta.get("artist", tagged_artist),
            )
        if additions:
            rows = pd.concat((rows, pd.DataFrame(additions)), ignore_index=True)
        return cls(rows, vectors, data_root, policy, user_paths, user_artists, user_titles)

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
        all_stems: bool = False,
    ) -> list[SearchResult]:
        """Return playable matches from the selected stem type or every type."""

        try:
            facets = self.policy.facets_for(stem)

        except ValueError as exc:
            raise SearchError(str(exc)) from exc

        if not facets:
            raise SearchError(f"No search facets are available for {stem} yet.")

        candidates = self.rows[self.rows["status"] == "ok"]
        if not all_stems:
            candidates = candidates[candidates["stem"] == stem]
        if candidates.empty:
            scope = "stems" if all_stems else f"{stem} stems"
            raise SearchError(f"No indexed {scope} are available.")
        if all_stems:
            # A future facet may not apply to every stem (for example melody
            # on drums). Cross-stem searches use only their shared facets.
            facets = tuple(facet for facet in facets if all(
                facet in self.policy.facets_for(candidate_stem)
                for candidate_stem in candidates["stem"].unique()
            ))
            if not facets:
                raise SearchError("No search facets are shared by the indexed stem types.")

        missing = [facet for facet in facets if facet not in query]
        if missing:
            raise SearchError(f"The query is missing: {', '.join(missing)}.")

        positions = candidates["row"].to_numpy(dtype=int)
        if (positions < 0).any() or any(
            (positions >= len(self.vectors[facet])).any() for facet in facets
        ):
            raise SearchError("The saved search index has invalid row numbers.")

        facet_scores = []
        for facet in facets:
            values = self.vectors[facet][positions]
            requested = np.asarray(query[facet], dtype=np.float32)
            if requested.shape != (values.shape[1],):
                raise SearchError(f"The {facet} query does not match the saved index.")
            if not np.isfinite(requested).all() or not np.any(requested):
                raise SearchError(f"The {facet} query has no usable search features.")
            blocks = (RHYTHM_BLOCKS if facet == config.RHYTHM and
                      values.shape[1] == config.RHYTHM_DIM else ((slice(None), 1.0),))
            candidate_parts, query_parts = [], []
            for region, contribution in blocks:
                block = values[:, region]
                mean = block.mean(axis=0)
                scale = block.std(axis=0) + 1e-8
                weight = np.sqrt(contribution / block.shape[1])
                candidate_parts.append((block - mean) / scale * weight)
                query_parts.append((requested[region] - mean) / scale * weight)
            matrix = np.hstack(candidate_parts)
            query_vector = np.concatenate(query_parts)
            query_norm = np.linalg.norm(query_vector)
            if not np.isfinite(query_norm) or query_norm < 1e-12:
                raise SearchError(f"The {facet} query has no usable search features.")
            norms = np.linalg.norm(matrix, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                score = (matrix @ query_vector) / (norms * query_norm)
            if facet == config.RHYTHM:
                score = np.where(np.any(values, axis=1), score, -np.inf)
            facet_scores.append(np.where(np.isfinite(score), score, -np.inf))

        # Every implemented facet contributes exactly one equally weighted score.
        scores = np.mean(facet_scores, axis=0)

        results = []
        for position in np.argsort(-scores, kind="stable"):
            if len(results) == limit or not np.isfinite(scores[position]):
                break
            row = candidates.iloc[int(position)]
            dataset, track_id = str(row["dataset"]), str(row["track_id"])
            matched_stem = str(row["stem"])
            mix_path = self._mix_path(dataset, track_id)
            stem_path = self.data_root / "stems" / dataset / track_id / f"{matched_stem}.flac"
            if mix_path is None or not mix_path.is_file() or not stem_path.is_file():
                continue
            if (
                upload is not None
                and ((
                    upload.clip_start_seconds == 0
                    and abs(upload.duration_seconds - CLIP_SECONDS) <= CLIP_TOLERANCE_SECONDS
                    and source_matches_file(mix_path, upload)
                ) or (
                    dataset == "user_added"
                    and upload.clip_bytes is not None
                    and bytes_match_file(mix_path, upload.clip_bytes)
                ))
            ):
                continue
            if dataset == "user_added":
                title = self.user_titles.get(track_id)
                artist = self.user_artists.get(track_id)
            else:
                tagged_title, tagged_artist = song_tags_from_media(mix_path)
                source_name = (f"{track_id.partition('_')[2]}{mix_path.suffix}"
                               if dataset == "fma" else mix_path.name)
                title, artist = song_identity(source_name, tagged_title, tagged_artist)
            results.append(SearchResult(dataset, track_id, matched_stem,
                                        float(scores[position]), mix_path, stem_path,
                                        artist, title))
        return results
