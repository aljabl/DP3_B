"""
Blind listening test comparing chroma, MFCC and combined stem descriptors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.neighbors import NearestNeighbors


class ListeningTest:
    """
    prepare() picks random query stems, retrieves the nearest stems with each descriptor
    and writes shuffled audio excerpts plus a rating sheet. The sheet does not say which
    descriptor returned which clip (that is in key.csv). score() reads the filled-in sheet.
    """

    DESCRIPTORS = {"chroma": ["chroma"], "mfcc": ["mfcc"], "combined": ["chroma", "mfcc"]}

    def __init__(self, index: pd.DataFrame, vectors: dict, stems_cache, out_dir: Path,
                 stems: tuple = ("drums", "bass", "vocals"), n_queries: int = 5,
                 top_k: int = 3, excerpt_s: float = 10.0, seed: int = 0):
        """
        index:   one row per stem with columns track_id, stem and row (position in vectors)
        vectors: {"chroma": array, "mfcc": array}
        """
        self.index = index
        self.vectors = vectors
        self.stems_cache = stems_cache
        self.out_dir = Path(out_dir)
        self.stems = stems
        self.n_queries = n_queries
        self.top_k = top_k
        self.excerpt_s = excerpt_s
        self.rng = np.random.default_rng(seed)

    def prepare(self):
        """Writes the excerpts, ratings.csv (to fill in) and key.csv (do not open before rating)."""

        self.out_dir.mkdir(parents=True, exist_ok=True)
        sheet, key = [], []

        for stem in self.stems:
            rows = self.index[self.index["stem"] == stem].reset_index(drop=True)
            embeddings = {name: self._embed(rows["row"], blocks)
                          for name, blocks in self.DESCRIPTORS.items()}
            queries = self.rng.choice(len(rows), self.n_queries, replace=False)

            for number, query in enumerate(queries, 1):
                found = self._retrieve(query, embeddings)
                self._export(rows.at[query, "track_id"], stem, f"{stem}_{number}_query.flac")

                for clip_number, position in enumerate(self.rng.permutation(list(found)), 1):
                    clip = f"{stem}_{number}_{clip_number}.flac"
                    track_id = rows.at[position, "track_id"]
                    self._export(track_id, stem, clip)
                    sheet.append({"stem": stem, "query": number, "clip": clip, "rating": ""})
                    key.append({"stem": stem, "query": number, "clip": clip,
                                "track_id": track_id, "returned_by": "|".join(found[position])})

        pd.DataFrame(sheet).to_csv(self.out_dir / "ratings.csv", index=False)
        pd.DataFrame(key).to_csv(self.out_dir / "key.csv", index=False)

    def score(self) -> pd.DataFrame:
        """
        Mean rating of each descriptor's results, and the number of queries it won or tied,
        per stem type. Queries left unrated are ignored.
        """

        ratings = pd.read_csv(self.out_dir / "ratings.csv").dropna(subset=["rating"])
        key = pd.read_csv(self.out_dir / "key.csv")

        rated = ratings.merge(key, on=["stem", "query", "clip"])
        rated["descriptor"] = rated["returned_by"].str.split("|")
        per_query = (rated.explode("descriptor")
                     .groupby(["stem", "query", "descriptor"])["rating"].mean().unstack())

        compared = per_query[list(self.DESCRIPTORS)]
        wins = compared.eq(compared.max(axis=1), axis=0).groupby("stem").sum()

        return pd.concat({"mean_rating": per_query.groupby("stem").mean(), "wins": wins}, axis=1)

    def _embed(self, rows, blocks: list) -> np.ndarray:
        """Standardizes each block, then weights it by 1/sqrt(dim) so that blocks count equally."""

        parts = []

        for block in blocks:
            x = self.vectors[block][rows]
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
            parts.append(x / np.sqrt(x.shape[1]))

        return np.hstack(parts)

    def _retrieve(self, query: int, embeddings: dict) -> dict:
        """Maps position -> descriptors that returned it, plus one random stem as a baseline."""

        found = {}

        for name, x in embeddings.items():
            nn = NearestNeighbors(n_neighbors=self.top_k + 1, metric="cosine").fit(x)
            neighbours = nn.kneighbors(x[[query]], return_distance=False)[0]

            for position in [n for n in neighbours if n != query][:self.top_k]:
                found.setdefault(position, []).append(name)

        n_stems = len(next(iter(embeddings.values())))
        random_position = self.rng.choice([i for i in range(n_stems) if i != query and i not in found])
        found[random_position] = ["random"]

        return found

    def _export(self, track_id: str, stem: str, filename: str):
        """Saves an excerpt from the middle of a stem."""

        source = self.stems_cache.paths(track_id)[stem]
        info = sf.info(source)
        length = int(self.excerpt_s * info.samplerate)
        start = max(0, (info.frames - length) // 2)

        audio, rate = sf.read(source, start=start, frames=length)
        sf.write(self.out_dir / filename, audio, rate)
