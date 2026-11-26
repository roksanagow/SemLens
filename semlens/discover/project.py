"""A discovery kept on disk, and the hand-off to an ordinary SemLens analysis.

Each discovery is a folder: the sample and its vectors (see
:mod:`.contextual`), ``discovery.json`` with the seeds, the marks and the
settings, ``seed_vectors.npz`` with the vectors of seed usages from outside
the sample, and the last ranking (``scores.npz``, and ``background.npz`` when
the background correction was used). Everything needed to continue exploring
is in the folder, so a reopened discovery shows its ranking at once.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data_loading import CorpusSentence, TargetWordData, UsageInstance, select_usages
from ..embeddings import embedded_usages_from_precomputed
from .contextual import VectorStore
from .seeds import SeedUsage, seed_rows

DISCOVERY_FILE = "discovery.json"


@dataclass
class Discovery:
    folder: Path
    name: str
    source: str = ""
    seed_words: list[str] = field(default_factory=list)
    seed_corpora: list[str] | None = None          # None: seeds from every corpus
    seed_usages: list[SeedUsage] = field(default_factory=list)
    marks: dict[int, bool] = field(default_factory=dict)        # sampled occurrence → fits the concept
    word_status: dict[str, str] = field(default_factory=dict)   # word → "accepted" or "rejected"
    add_marked: bool | None = None                 # whether marks and accepted words join the seeds (None, as in older files: no)
    settings: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def marked_rows(self) -> list[int]:
        return [row for row, fits in self.marks.items() if fits]

    @property
    def accepted_words(self) -> list[str]:
        return [w for w, status in self.word_status.items() if status == "accepted"]

    @property
    def rejected_words(self) -> list[str]:
        return [w for w, status in self.word_status.items() if status == "rejected"]

    def store(self) -> VectorStore:
        return VectorStore.load(self.folder)

    def save(self):
        content = {"name": self.name, "source": self.source, "seed_words": self.seed_words, "seed_corpora": self.seed_corpora,
                   "seed_usages": [u.as_dict() for u in self.seed_usages],
                   "marks": {str(r): bool(v) for r, v in self.marks.items()}, "word_status": self.word_status,
                   "add_marked": self.add_marked, "settings": self.settings, "created": self.created}
        _write_atomically(self.folder / DISCOVERY_FILE, json.dumps(content, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, folder: str | Path) -> "Discovery":
        folder = Path(folder)
        content = json.loads((folder / DISCOVERY_FILE).read_text())
        return cls(folder, content.get("name", folder.name), content.get("source", ""), list(content.get("seed_words", [])),
                   content.get("seed_corpora"), [SeedUsage(**u) for u in content.get("seed_usages", [])],
                   {int(r): bool(v) for r, v in content.get("marks", {}).items()}, dict(content.get("word_status", {})),
                   content.get("add_marked"), dict(content.get("settings", {})), float(content.get("created", 0) or 0))


def _write_atomically(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".saving-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def folder_name(name: str) -> str:
    """A folder name from a discovery's name: letters, digits, dots, hyphens and underscores."""
    safe = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    if not safe:
        raise ValueError("Give the discovery a name.")
    return safe[:80]


def list_discoveries(root: str | Path) -> list[dict]:
    """Saved discoveries, newest first: name, folder, model, corpora and number of occurrences."""
    root = Path(root)
    found = []
    if not root.is_dir():
        return found
    for folder in root.iterdir():
        if folder.name.startswith(".") or not (folder / DISCOVERY_FILE).is_file() or not (folder / "vectors.json").is_file():
            continue
        try:
            content = json.loads((folder / DISCOVERY_FILE).read_text())
            vectors = json.loads((folder / "vectors.json").read_text())
            corpora = json.loads((folder / "sample.json").read_text()).get("corpora", [])
        except (OSError, ValueError):
            continue
        found.append({"name": content.get("name", folder.name), "folder": str(folder), "created": content.get("created", 0),
                      "model": vectors.get("model_id", ""), "layer": vectors.get("layer"), "occurrences": vectors.get("occurrences", 0),
                      "representation": vectors.get("discover_representation", "word"), "corpora": corpora})
    return sorted(found, key=lambda d: -float(d["created"] or 0))


def delete_discovery(folder: str | Path):
    folder = Path(folder)
    if not (folder / DISCOVERY_FILE).is_file():
        raise ValueError("This folder is not a saved discovery.")
    shutil.rmtree(folder)


def load_seed_vectors(folder: str | Path) -> dict[str, np.ndarray]:
    path = Path(folder) / "seed_vectors.npz"
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as saved:
        return dict(zip(saved["keys"].tolist(), saved["vectors"]))


def save_seed_vectors(folder: str | Path, vectors: dict[str, np.ndarray]):
    path = Path(folder) / "seed_vectors.npz"
    if not vectors:
        path.unlink(missing_ok=True)
        return
    keys = list(vectors)
    np.savez(path, keys=np.array(keys), vectors=np.stack([vectors[k] for k in keys]).astype(np.float32))


SCORES_FILE = "scores.npz"
BACKGROUND_FILE = "background.npz"


def _save_arrays(path: Path, **arrays):
    fd, temp = tempfile.mkstemp(prefix=".saving-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_scoring(folder: str | Path, scoring):
    """Keep the last ranking's neighbours with the discovery, so that reopening it needs no new comparison."""
    keys = json.dumps([[kind, int(value) if kind == "row" else str(value)] for kind, value in scoring.keys])
    _save_arrays(Path(folder) / SCORES_FILE, distances=scoring.neighbours.distances, seeds=scoring.neighbours.seeds,
                 k=np.array(scoring.k), threshold=np.array(np.nan if scoring.threshold is None else scoring.threshold),
                 keys=np.array(keys))


def load_scoring(folder: str | Path):
    """The saved ranking, or None when there is none or it cannot be read."""
    from .knn import Neighbours, Scoring
    path = Path(folder) / SCORES_FILE
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            keys = tuple((kind, int(value) if kind == "row" else value) for kind, value in json.loads(saved["keys"].item()))
            threshold = float(saved["threshold"])
            return Scoring(keys, int(saved["k"]), Neighbours(saved["distances"], saved["seeds"]),
                           None if np.isnan(threshold) else threshold)
    except (OSError, ValueError, KeyError):
        return None


def save_background(folder: str | Path, k: int, neighbours):
    _save_arrays(Path(folder) / BACKGROUND_FILE, distances=neighbours.distances, seeds=neighbours.seeds, k=np.array(k))


def load_background(folder: str | Path):
    """(k, neighbours) of the saved background comparison, or None."""
    from .knn import Neighbours
    path = Path(folder) / BACKGROUND_FILE
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            return int(saved["k"]), Neighbours(saved["distances"], saved["seeds"])
    except (OSError, ValueError, KeyError):
        return None


# --- Hand-off to SemLens ----------------------------------------------------------------------------

def _records(sample) -> list[CorpusSentence]:
    return [CorpusSentence(c.corpus, c.text, f"discover-{i}") for i, c in enumerate(sample.contexts)]


def handoff_with_vectors(store: VectorStore, words, corpora=None):
    """The sampled occurrences of ``words`` with the vectors already made: a SemLens analysis at once.

    The vectors are the model's own (not centred), as SemLens would make them.
    """
    words = [w for w in dict.fromkeys(words) if w]
    if not words:
        raise ValueError("Choose at least one word.")
    sample = store.sample
    rows = seed_rows(sample, words, corpora, store.valid)
    if not len(rows):
        raise ValueError("None of these words has sampled usages in the chosen corpora.")
    usages = [UsageInstance(sample.text_of(r), words[0], *sample.span_of(r), sample.corpora[int(sample.corpus[r])],
                            f"discover-{int(sample.context[r])}") for r in rows]
    data = TargetWordData(words[0], usages, list(sample.corpora), words[1:])
    vectors = torch.from_numpy(np.asarray(store.vectors[rows], dtype=np.float32))
    metadata = {**store.metadata, "representation": "word", "pooling": "mean", "source": "Discover"}
    return data, embedded_usages_from_precomputed(data, vectors, metadata=metadata)


def handoff_sentences(sample, words, corpora=None) -> TargetWordData:
    """Every occurrence of ``words`` in the discovery's contexts, to embed in SemLens."""
    words = [w for w in dict.fromkeys(words) if w]
    if not words:
        raise ValueError("Choose at least one word.")
    records = [r for r in _records(sample) if corpora is None or r.corpus in set(corpora)]
    data = select_usages(records, words[0], words[1:])  # raises when none of the words occurs
    return TargetWordData(data.word, data.usages, [c for c in sample.corpora if c in data.corpus_labels], data.forms)


def source_texts(sample) -> dict[str, str]:
    """Each corpus's contexts, one paragraph each, for SemLens's other inputs."""
    texts: dict[str, list[str]] = {c: [] for c in sample.corpora}
    for context in sample.contexts:
        texts.setdefault(context.corpus, []).append(" ".join(context.text.split()))
    return {c: "\n\n".join(t) for c, t in texts.items()}
