"""Contextual vectors for sampled word occurrences, kept on disk.

Two representations. **Word vectors** (the default): every context holding a
sampled occurrence is read once by the model, and each occurrence's vector is
the mean of its own subword tokens (no special tokens, no target markers).
**Slot vectors**: the occurrence is replaced by the mask token and the vector
is read at the mask, so it describes what could fill the slot rather than the
word's spelling; this needs one pass per occurrence, so it is slower. Vectors are written to a 16-bit file as they are made, so
memory holds only the model and one batch. The mean of all vectors is saved
too: vectors are centred on it before comparison, because raw XLM-RoBERTa
vectors all point in much the same direction, which makes every pair look
similar under cosine distance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..embeddings import LoadedModel, extract_word_embeddings, iter_token_embeddings, model_metadata, pool_span
from .vocabulary import Sample

DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"
# Middle-to-upper layers usually carry word meaning better than the last one,
# which is shaped by the model's training task. Layer 8 of 12 for the base model.
DEFAULT_LAYER = 8


def default_layer(layers: int) -> int:
    """About two thirds of the way up: layer 8 of 12, layer 16 of 24."""
    return max(1, round(layers * 2 / 3))


@dataclass
class VectorStore:
    """A sample, its vectors on disk, and the mean they are centred on."""
    folder: Path
    sample: Sample
    vectors: np.ndarray        # memory-mapped, 16-bit, one row per sampled occurrence
    mean: np.ndarray
    valid: np.ndarray          # False where an occurrence could not be encoded
    metadata: dict

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def normalised(self, rows: slice | Sequence[int] | np.ndarray) -> np.ndarray:
        """Centred, unit-length 32-bit vectors for ``rows``."""
        block = np.asarray(self.vectors[rows], dtype=np.float32) - self.mean
        return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)

    @classmethod
    def load(cls, folder: str | Path) -> "VectorStore":
        folder = Path(folder)
        metadata = json.loads((folder / "vectors.json").read_text())
        return cls(folder, Sample.load(folder), np.load(folder / "vectors.npy", mmap_mode="r"), np.load(folder / "mean.npy"),
                   np.load(folder / "valid.npy"), metadata)


REPRESENTATIONS = ("word", "slot")


def masked_text(text: str, span: tuple[int, int], mask_token: str, window: int = 1200) -> str:
    """``text`` with the occurrence at ``span`` replaced by the mask token, cut to ``window`` characters each side."""
    start, end = span
    left = max(0, start - window)
    return text[left:start] + mask_token + text[end:end + window]


@torch.no_grad()
def iter_slot_vectors(loaded_model: LoadedModel, items, *, layer: int = DEFAULT_LAYER, max_length: int = 256,
                      batch_size: int = 16, progress_callback=None):
    """Yield (position, vector or None) for each (text, span): the vector at a mask put in place of the word.

    The vector describes the slot ("living in a ___ of yes-men"), what could
    fill it, rather than the word itself, so words spelled alike are not drawn
    together. Each occurrence needs its own pass through the model.
    """
    from ..embeddings import _hidden_states

    tokenizer, model, device = loaded_model.tokenizer, loaded_model.model, loaded_model.device
    if tokenizer.mask_token is None:
        raise ValueError("This model has no mask token, so slot vectors cannot be made. Choose word vectors, or a masked language model.")
    batches = (len(items) + batch_size - 1) // batch_size
    # About max_length / 2 tokens each side, so the mask is never cut off.
    window = max(200, max_length * 2)
    for b, start in enumerate(range(0, len(items), batch_size)):
        chunk = items[start:start + batch_size]
        texts = [masked_text(text, span, tokenizer.mask_token, window) for text, span in chunk]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        states = _hidden_states(model, encoded, layer)
        positions = encoded["input_ids"] == tokenizer.mask_token_id
        for i in range(len(chunk)):
            where = torch.nonzero(positions[i]).flatten()
            yield start + i, (states[i, where[0]].float().cpu().numpy() if len(where) == 1 else None)
        if progress_callback:
            progress_callback(b + 1, batches)


def encode_sample(loaded_model: LoadedModel, sample: Sample, folder: str | Path, *, layer: int = DEFAULT_LAYER,
                  max_length: int = 256, batch_size: int = 16, progress_callback=None, representation: str = "word") -> VectorStore:
    """Embed every sampled occurrence and save the vectors, the mean and the sample in ``folder``.

    ``representation`` is ``"word"`` (the word's own subword tokens, one pass
    per context) or ``"slot"`` (the word masked, one pass per occurrence).
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Representation must be one of {REPRESENTATIONS}.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if representation == "slot":
        return _encode_slots(loaded_model, sample, folder, layer=layer, max_length=max_length, batch_size=batch_size,
                             progress_callback=progress_callback)
    by_context: dict[int, list[int]] = {}
    for row, ctx in enumerate(sample.context.tolist()):
        by_context.setdefault(ctx, []).append(row)
    order = sorted(by_context)
    texts = [sample.contexts[c].text for c in order]
    dim = int(loaded_model.model.config.hidden_size)
    path = folder / "vectors.npy"
    vectors = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(sample), dim))
    valid = np.zeros(len(sample), dtype=bool)
    total = np.zeros(dim, dtype=np.float64)
    for i, offsets, token_vectors in iter_token_embeddings(loaded_model, texts, batch_size=batch_size, layer=layer,
                                                           max_length=max_length, progress_callback=progress_callback):
        text = texts[i]
        for row in by_context[order[i]]:
            try:
                vector = pool_span(token_vectors, offsets, sample.span_of(row), text).float().cpu().numpy()
            except ValueError:
                continue  # a word the tokenizer cannot cover (cut off by the context window, say)
            vectors[row] = vector
            valid[row] = True
            total += vector
    return _finish(loaded_model, sample, folder, vectors, valid, total, layer, max_length, "word")


def _finish(loaded_model, sample, folder, vectors, valid, total, layer, max_length, representation) -> VectorStore:
    vectors.flush()
    mean = (total / max(int(valid.sum()), 1)).astype(np.float32)
    np.save(folder / "mean.npy", mean)
    np.save(folder / "valid.npy", valid)
    metadata = {**model_metadata(loaded_model, layer=layer, max_length=max_length), "occurrences": len(sample),
                "encoded": int(valid.sum()), "discover_representation": representation}
    (folder / "vectors.json").write_text(json.dumps(metadata))
    sample.save(folder)
    del vectors
    return VectorStore.load(folder)


def _encode_slots(loaded_model, sample, folder, *, layer, max_length, batch_size, progress_callback) -> VectorStore:
    dim = int(loaded_model.model.config.hidden_size)
    vectors = np.lib.format.open_memmap(folder / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(sample), dim))
    valid = np.zeros(len(sample), dtype=bool)
    total = np.zeros(dim, dtype=np.float64)
    items = [(sample.text_of(r), sample.span_of(r)) for r in range(len(sample))]
    for row, vector in iter_slot_vectors(loaded_model, items, layer=layer, max_length=max_length, batch_size=batch_size,
                                         progress_callback=progress_callback):
        if vector is not None:
            vectors[row] = vector
            valid[row] = True
            total += vector
    return _finish(loaded_model, sample, folder, vectors, valid, total, layer, max_length, "slot")


def embed_seed_usage_dict(loaded_model: LoadedModel, usages, store: VectorStore, *, batch_size: int = 16) -> dict:
    """Vectors for seed usages (objects with ``key``, ``sentence``, ``start``, ``end``), keyed by ``key``.

    Usages that cannot be encoded are left out of the result.
    """
    usages = list(usages)
    found = {}
    for start in range(0, len(usages), 64):
        chunk = usages[start:start + 64]
        try:
            vectors = embed_seed_usages(loaded_model, [u.sentence for u in chunk], [(u.start, u.end) for u in chunk], store,
                                        batch_size=batch_size, strict=True)
            found.update(zip([u.key for u in chunk], vectors))
        except ValueError:
            for usage in chunk:
                try:
                    found[usage.key] = embed_seed_usages(loaded_model, [usage.sentence], [(usage.start, usage.end)], store, strict=True)[0]
                except ValueError:
                    continue
    return found


def embed_seed_usages(loaded_model: LoadedModel, sentences: Sequence[str], spans: Sequence[tuple[int, int]], store: VectorStore, *,
                      batch_size: int = 16, strict: bool = False) -> np.ndarray:
    """Vectors for seed usages given as sentences and target positions, in the store's space (centred, unit length).

    Usages that cannot be encoded (a word cut off by the context window) are
    left out, or with ``strict`` raise an error, so that rows stay aligned.
    """
    metadata = store.metadata
    layer, max_length = int(metadata.get("layer", DEFAULT_LAYER)), int(metadata.get("max_length", 256))
    if metadata.get("discover_representation") == "slot":
        found = [v for _, v in iter_slot_vectors(loaded_model, list(zip(sentences, [tuple(s) for s in spans])), layer=layer,
                                                 max_length=max_length, batch_size=batch_size)]
        if strict and any(v is None for v in found):
            raise ValueError("A seed usage could not be encoded.")
        found = [v for v in found if v is not None]
        if not found:
            raise ValueError("None of the seed usages could be encoded.")
        vectors = np.stack(found) - store.mean
    else:
        vectors = extract_word_embeddings(loaded_model, list(sentences), [tuple(s) for s in spans], batch_size=batch_size,
                                          max_length=max_length, layer=layer, target_markers=False).float().cpu().numpy() - store.mean
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)



def contexts_from_cache(cache, restrict: str | None = None):
    """The sentences of a saved corpus collection as contexts, restricted like :func:`contexts_from_texts`."""
    from ..utils import usage_search_pattern
    from .vocabulary import Context

    pattern = usage_search_pattern(restrict) if restrict and restrict.strip() else None
    return [Context(corpus, sentence) for _, corpus, sentence, _, _ in cache.rows if pattern is None or pattern.search(sentence)]


def encode_from_cache(cache, sample: Sample, folder: str | Path, *, progress_callback=None) -> VectorStore:
    """Vectors for the sampled occurrences pooled from a saved corpus collection's token vectors: no model needed.

    The sample's contexts must be the collection's sentences (see :func:`contexts_from_cache`).
    """
    if cache.offsets is None:
        raise ValueError("This saved collection holds sentence vectors only; word vectors need a collection saved with word representations.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    where = {(corpus, sentence): (start, count) for _, corpus, sentence, start, count in cache.rows}
    dim = int(cache.vectors.shape[1])
    vectors = np.lib.format.open_memmap(folder / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(sample), dim))
    valid = np.zeros(len(sample), dtype=bool)
    total = np.zeros(dim, dtype=np.float64)
    step = max(1, len(sample) // 100)
    for row in range(len(sample)):
        context = sample.contexts[int(sample.context[row])]
        found = where.get((context.corpus, context.text))
        if found is not None:
            start, count = found
            try:
                vector = pool_span(torch.from_numpy(np.array(cache.vectors[start:start + count], dtype=np.float32)),
                                   np.asarray(cache.offsets[start:start + count]), sample.span_of(row), context.text).numpy()
            except ValueError:
                vector = None
            if vector is not None:
                vectors[row] = vector
                valid[row] = True
                total += vector
        if progress_callback and (row % step == 0 or row == len(sample) - 1):
            progress_callback(row + 1, len(sample))
    vectors.flush()
    mean = (total / max(int(valid.sum()), 1)).astype(np.float32)
    np.save(folder / "mean.npy", mean)
    np.save(folder / "valid.npy", valid)
    metadata = {**cache.info["model"], "occurrences": len(sample), "encoded": int(valid.sum()), "discover_representation": "word",
                "from_saved_corpora": cache.info.get("name", ""), "cache_id": cache.path.name}
    (folder / "vectors.json").write_text(json.dumps(metadata))
    sample.save(folder)
    del vectors
    return VectorStore.load(folder)
