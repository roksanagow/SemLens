"""Embed the packaged example once, so that it opens without a model download.

Writes ``app/data/bubble_embeddings.npz``: the XL-LEXEME embeddings of the
example's 350 usages in 16-bit, with the usage IDs they belong to, and four
definitions with their embeddings. Run again with the project's Python
environment whenever the example's sentences or SemLens's default embedding
settings change:

    .venv/bin/python scripts/build_example_embeddings.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semlens.data_loading import load_from_csv
from semlens.definitions import embed_definitions, format_definitions
from semlens.embeddings import embed_usages, load_model

MODEL = "pierluigic/xl-lexeme"
DEFINITIONS = [
    "A small pocket of gas enclosed in liquid or a thin film.",
    "A situation in which asset prices rise beyond a sustainable value.",
    "A restricted group of people permitted close contact during pandemic restrictions.",
    "An insulated social or informational environment whose members are cut off from outside views.",
]


def main():
    csv = ROOT / "app" / "data" / "bubble_usages.csv"
    data = load_from_csv(csv, "bubble", forms=["bubbles"], time_col="date")
    model = load_model(MODEL, device="cpu")
    # SemLens's defaults: the target word, the final layer, 256-token windows, XL-LEXEME's target markers.
    embedded = embed_usages(model, data, layer=-1, max_length=256, batch_size=16, representation="word")
    formatted = format_definitions(data.word, DEFINITIONS)
    markers = bool(embedded.metadata.get("target_markers"))
    definition_vectors = embed_definitions(model, data.word, formatted, layer=-1, template="{word}: {definition}",
                                           readout="word", max_length=256, target_markers=markers)
    manifest = {"embedding_metadata": embedded.metadata, "definitions_raw": DEFINITIONS, "definitions_formatted": formatted,
                "definition_settings": {"template": "{word}: {definition}", "readout": "word", "model_id": model.name,
                                        "layer": -1, "target_markers": markers}}
    output = ROOT / "app" / "data" / "bubble_embeddings.npz"
    np.savez_compressed(output, vectors=embedded.embeddings.numpy().astype(np.float16),
                        usage_ids=np.array([u.usage_id for u in data.usages]),
                        definition_vectors=definition_vectors.numpy().astype(np.float32),
                        manifest=np.array(json.dumps(manifest, ensure_ascii=False)))
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.2f} MB): {len(data)} usages, {len(formatted)} definitions")


if __name__ == "__main__":
    main()
