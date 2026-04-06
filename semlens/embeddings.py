"""Contextualised word embedding extraction using HuggingFace encoders.

This module is deliberately decoupled from the rest of the tool so that
advanced users can substitute their own embedding pipeline.  The only
requirement is that you produce a ``torch.Tensor`` of shape ``[N, D]``
aligned with the usages in a ``TargetWordData`` object.

Quick-start
-----------
>>> model = load_model("pierluigic/xl-lexeme")
>>> data  = load_from_sentences(...)
>>> embs  = embed_usages(model, data)  # Tensor[N, D]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .data_loading import TargetWordData

# ---------------------------------------------------------------------------
# Short-name aliases for common models (convenience, not required)
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    # Specialised
    "xl-lexeme": "pierluigic/xl-lexeme",
    # Multilingual
    "xlm-roberta": "FacebookAI/xlm-roberta-large",
    "multilingual-e5": "intfloat/multilingual-e5-large",
    "rembert": "google/rembert",
    "mmbert": "jhu-clsp/mmBERT-base",
    # English monolingual
    "roberta": "FacebookAI/roberta-large",
}


def resolve_model_name(name: str) -> str:
    """Resolve a short alias to a HuggingFace model identifier."""
    return MODEL_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Container for a tokeniser/model pair with device info."""
    tokenizer: object
    model: object
    device: torch.device
    name: str

    def as_tuple(self) -> tuple:
        """Return (tokenizer, model) for compatibility."""
        return (self.tokenizer, self.model)


def load_model(
    model_name: str,
    device: str | torch.device | None = None,
) -> LoadedModel:
    """Load a HuggingFace encoder model and its tokeniser.

    Parameters
    ----------
    model_name : str
        Either a short alias (see ``MODEL_ALIASES``) or a full HuggingFace
        model identifier / local path.
    device : str or torch.device, optional
        Target device.  Auto-detected if not given (CUDA → MPS → CPU).

    Returns
    -------
    LoadedModel
        Contains ``.tokenizer``, ``.model``, ``.device``, and ``.name``.
    """
    from transformers import AutoTokenizer, AutoModel

    resolved = resolve_model_name(model_name)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    tok = AutoTokenizer.from_pretrained(resolved, use_fast=True)
    mdl = AutoModel.from_pretrained(resolved)
    mdl.to(device)
    mdl.eval()

    return LoadedModel(tokenizer=tok, model=mdl, device=device, name=resolved)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

CharSpan = Union[Tuple[int, int], List[int]]


@torch.no_grad()
def extract_word_embeddings(
    loaded_model: LoadedModel,
    sentences: Sequence[str],
    char_spans: Sequence[CharSpan],
    *,
    batch_size: int = 32,
    pooling: str = "mean",
    max_length: int | None = 256,
    progress_callback=None,
) -> Tensor:
    """Extract contextualised embeddings for the target word in each sentence.

    For each sentence, the target word's character span is mapped to
    sub-word tokens via the fast tokeniser's offset mapping.  The hidden
    states of those tokens are pooled (mean or first) to produce one
    embedding vector per usage.

    Parameters
    ----------
    loaded_model : LoadedModel
        From ``load_model()``.
    sentences : list[str]
        The usage sentences.
    char_spans : list of (start, end)
        Character-level spans of the target word in each sentence.
    batch_size : int
        Inference batch size.
    pooling : str
        ``"mean"`` (default) or ``"first"`` sub-word token.
    max_length : int or None
        Maximum token length for truncation.
    progress_callback : callable, optional
        Called with ``(batch_index, total_batches)`` after each batch.
        Useful for Streamlit progress bars.

    Returns
    -------
    Tensor
        Shape ``[len(sentences), hidden_dim]``.
    """
    tok = loaded_model.tokenizer
    mdl = loaded_model.model
    dev = loaded_model.device

    spans = [(int(s[0]), int(s[1])) for s in char_spans]
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    out: list[Tensor] = []

    for batch_idx in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_idx : batch_idx + batch_size]
        batch_spans = spans[batch_idx : batch_idx + batch_size]

        enc = tok(
            list(batch_sents),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        offsets = enc.pop("offset_mapping")  # [B, T, 2] — stays on CPU
        enc = {k: v.to(dev) for k, v in enc.items()}

        hidden = mdl(**enc).last_hidden_state  # [B, T, H]
        attn = enc.get("attention_mask", None)

        for b in range(hidden.size(0)):
            start, end = batch_spans[b]
            off = offsets[b].tolist()

            # Find sub-word tokens overlapping the character span
            token_ids: list[int] = []
            for t, (s_off, e_off) in enumerate(off):
                if s_off == 0 and e_off == 0:
                    continue  # skip special tokens
                if e_off <= start or s_off >= end:
                    continue
                token_ids.append(t)

            # Fallback: closest token by start position
            if not token_ids:
                best_t, best_dist = None, float("inf")
                for t, (s_off, e_off) in enumerate(off):
                    if s_off == 0 and e_off == 0:
                        continue
                    d = min(abs(s_off - start), abs(e_off - start))
                    if d < best_dist:
                        best_dist = d
                        best_t = t
                token_ids = [best_t] if best_t is not None else [0]

            ids_t = torch.tensor(token_ids, device=dev, dtype=torch.long)
            vecs = hidden[b].index_select(0, ids_t)  # [K, H]

            if attn is not None:
                mask = attn[b].index_select(0, ids_t).unsqueeze(-1).float()
                vecs = vecs * mask
                denom = mask.sum().clamp_min(1.0)
            else:
                denom = torch.tensor(vecs.size(0), device=dev, dtype=torch.float)

            if pooling == "first":
                emb = vecs[0]
            elif pooling == "mean":
                emb = vecs.sum(dim=0) / denom
            else:
                raise ValueError("pooling must be 'mean' or 'first'")

            out.append(emb.detach().cpu())

        if progress_callback is not None:
            progress_callback(batch_idx // batch_size + 1, total_batches)

    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedUsages:
    """Embeddings for all usages of a target word, with corpus metadata."""

    word: str
    usages: list  # list[UsageInstance]
    embeddings: Tensor  # [N, D]
    corpus_labels: list[str]  # unique sorted corpus names
    _corpus_indices: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self._corpus_indices:
            for i, u in enumerate(self.usages):
                self._corpus_indices.setdefault(u.corpus, []).append(i)

    def get_corpus_embeddings(self, corpus: str) -> Tensor:
        """Return embeddings for a single corpus."""
        idx = self._corpus_indices[corpus]
        return self.embeddings[idx]

    def get_corpus_indices(self, corpus: str) -> list[int]:
        return self._corpus_indices.get(corpus, [])

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]


def embed_usages(
    loaded_model: LoadedModel,
    data: TargetWordData,
    *,
    batch_size: int = 32,
    pooling: str = "mean",
    max_length: int | None = 256,
    progress_callback=None,
) -> EmbeddedUsages:
    """Embed all usages from a ``TargetWordData`` object.

    This is the main entry point for most users.
    """
    embs = extract_word_embeddings(
        loaded_model,
        data.sentences,
        data.char_spans,
        batch_size=batch_size,
        pooling=pooling,
        max_length=max_length,
        progress_callback=progress_callback,
    )
    return EmbeddedUsages(
        word=data.word,
        usages=data.usages,
        embeddings=embs,
        corpus_labels=data.corpus_labels,
    )


def embedded_usages_from_precomputed(
    data: TargetWordData,
    embeddings: Tensor,
) -> EmbeddedUsages:
    """Create ``EmbeddedUsages`` from pre-computed embeddings.

    The embeddings must be aligned with ``data.usages`` (same order, same
    count).
    """
    if embeddings.shape[0] != len(data):
        raise ValueError(
            f"Embedding count ({embeddings.shape[0]}) does not match "
            f"usage count ({len(data)})."
        )
    return EmbeddedUsages(
        word=data.word,
        usages=data.usages,
        embeddings=embeddings,
        corpus_labels=data.corpus_labels,
    )
