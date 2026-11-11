"""Encoder representations with explicit layers and safe target alignment.

Long sentences are encoded in overlapping windows. Offset mappings refer to
stored sentence text; a target is never replaced by a nearby token.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import Tensor
from .data_loading import TargetWordData, validate_embeddings

MODEL_ALIASES = {
    "xl-lexeme": "pierluigic/xl-lexeme",
    "xlm-roberta": "FacebookAI/xlm-roberta-large",
    "multilingual-e5": "intfloat/multilingual-e5-large",
    "rembert": "google/rembert", "mmbert": "jhu-clsp/mmBERT-base",
    "roberta": "FacebookAI/roberta-large",
}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    device: torch.device
    name: str

    def as_tuple(self):
        return self.tokenizer, self.model


def load_model(model_name: str, device=None, *, revision: str | None = None, local_files_only=False) -> LoadedModel:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    resolved = resolve_model_name(model_name)
    source = resolved
    if local_files_only and not Path(resolved).is_dir():
        # Use a real local path so tokenizer initialisation cannot perform
        # auxiliary Hub lookups despite local_files_only=True.
        from huggingface_hub import snapshot_download
        source = snapshot_download(resolved, revision=revision, local_files_only=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    kwargs = {"local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(source, **kwargs)
    architectures = getattr(config, "architectures", None) or []
    if config.is_encoder_decoder or config.is_decoder or any("CausalLM" in a or "LMHeadModel" in a for a in architectures):
        raise ValueError("Choose an encoder model. Decoder and encoder-decoder extraction will be added after evaluation.")
    tok = AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
    if not tok.is_fast:
        raise ValueError("This model needs a fast tokenizer with character offsets.")
    mdl = AutoModel.from_pretrained(source, config=config, **kwargs)
    mdl.to(device).eval()
    return LoadedModel(tok, mdl, device, resolved)


def model_metadata(loaded: LoadedModel, *, layer=-1, representation="word", pooling="mean", max_length=256):
    config = loaded.model.config
    return {"model_id": loaded.name, "revision": getattr(config, "_commit_hash", None),
            "layer": int(layer), "representation": representation, "pooling": pooling,
            "max_length": int(max_length), "window_strategy": "overlap_first_occurrence_v1",
            "tokenizer": type(loaded.tokenizer).__name__,
            "dimension": int(getattr(config, "hidden_size", getattr(config, "d_model", 0))),
            "hidden_state_source": "last_hidden_state" if layer == -1 else f"hidden_states[{layer}]"}


def _hidden_states(model, inputs, layer):
    if layer == -1:
        return model(**inputs).last_hidden_state
    if layer < 0:
        raise ValueError("Layer must be -1 (final output), 0 (input embeddings), or a positive layer index.")
    outputs = model(**inputs, output_hidden_states=True)
    if outputs.hidden_states is None or layer >= len(outputs.hidden_states):
        raise ValueError(f"Layer {layer} is unavailable for this model.")
    return outputs.hidden_states[layer]


@torch.no_grad()
def iter_token_embeddings(loaded_model: LoadedModel, sentences: Sequence[str], *, batch_size=16, layer=-1,
                          max_length=256, progress_callback=None):
    """Yield (sentence index, offsets [tokens,2], embeddings [tokens,D]).

    Repeated tokens in overlapping windows use their first encoded occurrence.
    Padding and special tokens are excluded, so both word and sentence pooling
    can reuse the same saved vectors.
    """
    if batch_size < 1 or not sentences:
        raise ValueError("Provide sentences and a positive batch size.")
    tok, mdl, dev = loaded_model.tokenizer, loaded_model.model, loaded_model.device
    model_limit = getattr(tok, "model_max_length", 512)
    limit = min(max_length or 256, model_limit if model_limit < 100000 else 512)
    if limit < 8:
        raise ValueError("The context window must contain at least 8 tokens.")
    stride = min(32, max(0, (limit - tok.num_special_tokens_to_add()) // 4))
    total = (len(sentences) + batch_size - 1) // batch_size
    for start in range(0, len(sentences), batch_size):
        batch = list(sentences[start:start+batch_size])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=limit,
                  stride=stride, return_offsets_mapping=True, return_overflowing_tokens=True)
        offsets = enc.pop("offset_mapping")
        mapping = enc.pop("overflow_to_sample_mapping").tolist()
        by_sentence = [dict() for _ in batch]
        for chunk in range(0, len(mapping), batch_size):
            inputs = {k: v[chunk:chunk+batch_size].to(dev) for k, v in enc.items()}
            hidden = _hidden_states(mdl, inputs, layer).detach().cpu().float()
            for b in range(len(hidden)):
                ci = chunk+b
                target = by_sentence[mapping[ci]]
                repeated = {}
                for t, (s,e) in enumerate(offsets[ci].tolist()):
                    if s == e or not enc["attention_mask"][ci,t]:
                        continue
                    base = (s, e, int(enc["input_ids"][ci,t]))
                    ordinal = repeated.get(base, 0)
                    repeated[base] = ordinal + 1
                    target.setdefault((*base, ordinal), hidden[b,t].clone())
        for b, tokens in enumerate(by_sentence):
            if not tokens:
                raise ValueError(f"Sentence {start+b+1} has no encodable text.")
            keys = sorted(tokens, key=lambda key: key[:2])
            yield start+b, np.asarray([k[:2] for k in keys], dtype=np.int64), torch.stack([tokens[k] for k in keys])
        if progress_callback:
            progress_callback(start//batch_size+1, total)


def pool_span(vectors: Tensor, offsets: np.ndarray, span: tuple[int,int], sentence: str, pooling="mean") -> Tensor:
    start,end = span
    if not 0 <= start < end <= len(sentence):
        raise ValueError(f"Invalid target span {span}.")
    selected = np.flatnonzero((offsets[:,1] > start) & (offsets[:,0] < end))
    if not len(selected):
        raise ValueError(f"Target {sentence[start:end]!r} has no encoded tokens. Check the supplied offsets.")
    # Detect missing target pieces, including partial truncation, instead of silently pooling a fragment.
    covered = np.zeros(end-start, dtype=bool)
    for s,e in offsets[selected]:
        covered[max(s,start)-start:min(e,end)-start] = True
    if any(not covered[i] and not ch.isspace() for i,ch in enumerate(sentence[start:end])):
        raise ValueError(f"Target {sentence[start:end]!r} is only partly covered by the tokenizer.")
    if pooling == "first":
        return vectors[int(selected[0])]
    if pooling != "mean":
        raise ValueError("Pooling must be mean or first.")
    return vectors[selected.tolist()].mean(0)


def extract_word_embeddings(loaded_model: LoadedModel, sentences: Sequence[str], char_spans: Sequence,
                            *, batch_size=16, pooling="mean", max_length=256, progress_callback=None,
                            layer=-1, representation="word") -> Tensor:
    if len(sentences) != len(char_spans):
        raise ValueError("Each sentence needs a corresponding target span.")
    if representation not in {"word", "sentence"}:
        raise ValueError("Representation must be word or sentence.")
    # Encode each unique context once, even when it contains several targets.
    unique = list(dict.fromkeys(sentences))
    tokens = {}
    for i, offsets, vectors in iter_token_embeddings(loaded_model, unique, batch_size=batch_size, layer=layer,
                                                    max_length=max_length, progress_callback=progress_callback):
        tokens[unique[i]] = (offsets, vectors)
    out = []
    for sentence, span in zip(sentences, char_spans):
        offsets,vectors = tokens[sentence]
        out.append(vectors.mean(0) if representation == "sentence" else pool_span(vectors,offsets,span,sentence,pooling))
    return torch.stack(out)


@dataclass
class EmbeddedUsages:
    word: str
    usages: list
    embeddings: Tensor
    corpus_labels: list[str]
    _corpus_indices: dict[str,list[int]] = field(default_factory=dict, repr=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.embeddings = validate_embeddings(self.embeddings, len(self.usages))
        self._corpus_indices = {}
        for i,u in enumerate(self.usages):
            self._corpus_indices.setdefault(u.corpus, []).append(i)

    def get_corpus_embeddings(self, corpus: str):
        return self.embeddings[self._corpus_indices[corpus]]

    def get_corpus_indices(self, corpus: str):
        return self._corpus_indices.get(corpus, [])

    @property
    def dim(self):
        return self.embeddings.shape[1]


def embed_usages(loaded_model: LoadedModel, data: TargetWordData, *, batch_size=16, pooling="mean", max_length=256,
                 progress_callback=None, layer=-1, representation="word") -> EmbeddedUsages:
    embs=extract_word_embeddings(loaded_model,data.sentences,data.char_spans,batch_size=batch_size,pooling=pooling,
                                 max_length=max_length,progress_callback=progress_callback,layer=layer,representation=representation)
    return EmbeddedUsages(data.word,data.usages,embs,data.corpus_labels,
                          metadata=model_metadata(loaded_model,layer=layer,representation=representation,pooling=pooling,max_length=max_length))


def embedded_usages_from_precomputed(data: TargetWordData, embeddings: Tensor, *, metadata=None) -> EmbeddedUsages:
    return EmbeddedUsages(data.word,data.usages,embeddings,data.corpus_labels,metadata=metadata or {})
