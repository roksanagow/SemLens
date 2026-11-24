"""Estimates of what heavy work costs, and whether this computer can afford it.

Every heavy action (downloading a model, embedding, saving corpus vectors,
comparing large corpora) builds a :class:`Cost` and asks :func:`assess` for a
verdict against the computer's free disk and memory:

- ``ok``: go ahead;
- ``confirm``: large enough that the user should confirm first;
- ``block``: it would fill the disk or exceed the computer's memory, which
  can freeze the computer, so it is not started.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

GB = 1e9
# Thresholds agreed for SemLens.
DISK_RESERVE = 5 * GB          # never leave less free disk than this
MEMORY_BLOCK_SHARE = 0.75      # never need more than this share of the RAM
CONFIRM_DISK = 2 * GB
CONFIRM_MEMORY_SHARE = 0.40
CONFIRM_SECONDS = 600
# Rough speeds for estimates before a model is measured: useful operations per
# second, per kind of device. Two operations per parameter per token.
ROUGH_FLOPS = {"cuda": 5e12, "mps": 0.8e12, "cpu": 0.15e12}  # mps measured: XL-LEXEME, M-series laptop
BYTES_PER_PAIR = 16            # a distance and its sort index, per pair of usages (SAMD)


@dataclass
class Resources:
    total_memory: float
    available_memory: float
    free_disk: float


def system_resources(path: str | Path = ".") -> Resources:
    """Memory of this computer and free disk where files will be written."""
    import psutil

    memory = psutil.virtual_memory()
    folder = Path(path).expanduser()
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return Resources(float(memory.total), float(memory.available), float(shutil.disk_usage(folder).free))


@dataclass
class Cost:
    """What an action needs. Bytes are estimates; ``seconds`` is None when unknown."""
    action: str
    download_bytes: float = 0.0
    disk_bytes: float = 0.0
    memory_bytes: float = 0.0
    seconds: float | None = None
    rough_time: bool = True
    details: list[str] = field(default_factory=list)

    @property
    def total_disk(self):
        return self.download_bytes + self.disk_bytes


@dataclass
class Verdict:
    level: str                      # "ok", "confirm" or "block"
    reasons: list[str]

    @property
    def allowed(self):
        return self.level != "block"


def assess(cost: Cost, resources: Resources) -> Verdict:
    """Whether the computer can afford ``cost``, with the reasons for any warning."""
    block, confirm = [], []
    if cost.total_disk and resources.free_disk - cost.total_disk < DISK_RESERVE:
        block.append(f"It needs about {format_bytes(cost.total_disk)} of disk space, and only {format_bytes(resources.free_disk)} is free: "
                     f"SemLens keeps at least {format_bytes(DISK_RESERVE)} free so the computer keeps working.")
    if cost.memory_bytes > MEMORY_BLOCK_SHARE * resources.total_memory:
        block.append(f"It needs about {format_bytes(cost.memory_bytes)} of memory, more than {MEMORY_BLOCK_SHARE:.0%} of this computer’s "
                     f"{format_bytes(resources.total_memory)}. That can freeze the computer.")
    if block:
        return Verdict("block", block)
    if cost.total_disk > CONFIRM_DISK:
        confirm.append(f"It uses about {format_bytes(cost.total_disk)} of disk space ({format_bytes(resources.free_disk)} is free).")
    if cost.memory_bytes > CONFIRM_MEMORY_SHARE * resources.total_memory:
        confirm.append(f"It needs about {format_bytes(cost.memory_bytes)} of memory, {cost.memory_bytes / resources.total_memory:.0%} of this computer’s. "
                       "Close other programs first.")
    if cost.seconds is not None and cost.seconds > CONFIRM_SECONDS:
        confirm.append(f"It may take {format_duration(cost.seconds)}{' (a rough estimate)' if cost.rough_time else ''}.")
    return Verdict("confirm" if confirm else "ok", confirm)


def format_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    minutes = seconds / 60
    if minutes < 60:
        return f"about {math.ceil(minutes)} minute{'s' if math.ceil(minutes) != 1 else ''}"
    hours = minutes / 60
    return f"about {hours:.1f} hours" if hours < 10 else f"about {round(hours)} hours"


# --- Estimates for embedding ----------------------------------------------------------------------

def rough_tokens_per_second(parameters: float, device: str) -> float:
    """A rough speed before measuring: about two operations per parameter per token."""
    return ROUGH_FLOPS.get(device, ROUGH_FLOPS["cpu"]) / max(2 * parameters, 1.0)


def count_tokens(texts, tokenizer=None, sample: int = 2000) -> int:
    """Tokens in ``texts``: counted on a sample with the tokenizer, or about 1.4 per word without one."""
    texts = list(texts)
    if not texts:
        return 0
    step = max(1, len(texts) // sample)
    chosen = texts[::step]
    if tokenizer is None:
        per_text = sum(len(t.split()) for t in chosen) * 1.4 / len(chosen)
    else:
        counted = 0
        for start in range(0, len(chosen), 64):
            encoded = tokenizer(chosen[start:start + 64], add_special_tokens=True, truncation=False, return_attention_mask=False)
            counted += sum(len(ids) for ids in encoded["input_ids"])
        per_text = counted / len(chosen)
    return int(round(per_text * len(texts)))


def model_memory(parameters: float, bytes_per_parameter: int = 4) -> float:
    """Memory for a model's weights while it runs (32-bit)."""
    return parameters * bytes_per_parameter


def activation_memory(batch_size: int, max_length: int, hidden: int, layers: int) -> float:
    """Memory for one batch of hidden states from every layer (32-bit)."""
    return batch_size * max_length * hidden * (layers + 1) * 4.0


def embedding_cost(action: str, *, texts, tokenizer=None, parameters: float, hidden: int, layers: int, device: str,
                   vectors: int, batch_size: int = 16, max_length: int = 256, tokens_per_second: float | None = None,
                   download_bytes: float = 0.0, disk_bytes: float = 0.0, vector_bytes: int = 4) -> Cost:
    """Time, memory and disk for running a model over ``texts`` and keeping ``vectors`` vectors."""
    tokens = count_tokens(texts, tokenizer)
    speed = tokens_per_second or rough_tokens_per_second(parameters, device)
    memory = model_memory(parameters) + activation_memory(batch_size, max_length, hidden, layers) + vectors * hidden * vector_bytes
    details = [f"{len(texts) if hasattr(texts, '__len__') else 'all'} contexts, about {tokens:,} tokens"
               + (f"; {vectors:,} vectors to keep" if vectors else "")]
    return Cost(action, download_bytes=download_bytes, disk_bytes=disk_bytes, memory_bytes=memory,
                seconds=tokens / speed if speed else None, rough_time=tokens_per_second is None, details=details)


def measure_tokens_per_second(loaded_model, texts, *, layer=-1, batch_size=16, max_length=256, seconds=5.0) -> float:
    """Tokens per second for this model on this computer, from at most ``seconds`` of real work."""
    import torch

    texts = [t for t in texts if t.strip()] or ["A short sentence to time the model."]
    tokenizer, model, device = loaded_model.tokenizer, loaded_model.model, loaded_model.device
    started, tokens, position = time.perf_counter(), 0, 0
    with torch.no_grad():
        while True:
            batch = [texts[(position + i) % len(texts)] for i in range(batch_size)]
            position += batch_size
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            model(**encoded, output_hidden_states=layer != -1)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            tokens += int(encoded["attention_mask"].sum())
            elapsed = time.perf_counter() - started
            if elapsed >= seconds or position >= 8 * len(texts) and elapsed > 0.5:
                return tokens / max(elapsed, 1e-6)


# --- Estimates for comparing corpora --------------------------------------------------------------

def comparison_cost(sizes: list[tuple[int, int]], action: str = "Comparing corpora") -> Cost:
    """Peak memory of the largest pairwise comparison among ``sizes`` (SAMD sorts every pair)."""
    largest = max((a * b for a, b in sizes), default=0)
    return Cost(action, memory_bytes=largest * BYTES_PER_PAIR,
                details=[f"The largest comparison is {max(sizes, key=lambda p: p[0] * p[1])[0]:,} × "
                         f"{max(sizes, key=lambda p: p[0] * p[1])[1]:,} usages"] if sizes else [])
