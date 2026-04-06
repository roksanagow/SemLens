# SemLens 🔍

**An Interactive Tool for Interpretable Semantic Change Analysis via Definition-Aligned Embedding Spaces**

SemLens is a Python library and Streamlit web application for visualising and quantifying lexical semantic change using contextualised embeddings, dictionary-informed definition spaces, and the four LSCD metrics from [Goworek & Dubossarsky (2026)](https://aclanthology.org/2026.lchange-1.13.pdf).

## Features

- **Four LSCD metrics**: APD, PRT, AMD, SAMD — plus directional AMD for asymmetric change detection (sense broadening vs. narrowing)
- **Definition-space projection**: project usage embeddings onto dictionary definitions to create an interpretable low-dimensional semantic space
- **Interactive 2D scatter plots**: hover to read sentences, colour-coded by corpus, with PCA / UMAP / t-SNE projection
- **LDA visualisation**: Linear Discriminant Analysis to find the axis maximally separating corpora, with interpretable definition weights
- **Per-definition metric breakdown**: see which dictionary senses drive the biggest distributional shift
- **Sense annotation**: click-to-annotate usages into named sense classes and export annotated datasets
- **Wiktionary integration**: fetch definitions automatically via the MediaWiki API (multilingual)
- **Flexible input**: paste sentences, upload CSV/TSV, or load pre-computed embeddings
- **Colourblind-friendly**: multiple accessible colour palettes (Wong, Tol Bright, IBM Design, etc.)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/semlens.git
cd semlens

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install the package with all dependencies
pip install -e ".[all]"
```

### Dependencies

- Python ≥ 3.10
- PyTorch, Transformers, scikit-learn, pandas, NumPy
- Streamlit, Plotly (for the web app)
- umap-learn (optional, for UMAP projections)

## Quick Start

### Web application

```bash
source .venv/bin/activate        # if not already active
streamlit run app/app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

### As a Python library

```python
from semlens.data_loading import load_from_sentences
from semlens.embeddings import load_model, embed_usages
from semlens.metrics import compute_all_metrics
from semlens.spaces import project_to_definition_space
from semlens.definitions import format_definitions, embed_definitions

# 1. Load data
data = load_from_sentences(
    sentences=["I deposited money at the bank", "The river bank was muddy", ...],
    word="bank",
    corpus_labels=["modern", "modern", "old", ...],
)

# 2. Embed usages
model = load_model("xl-lexeme")
embedded = embed_usages(model, data)

# 3. Compute metrics
embs_old = embedded.get_corpus_embeddings("old")
embs_new = embedded.get_corpus_embeddings("modern")
metrics = compute_all_metrics(embs_old, embs_new, ("old", "modern"))
print(metrics)

# 4. Definition space (optional)
defs = format_definitions("bank", [
    "a financial institution",
    "the side of a river or lake",
])
def_embs = embed_definitions(model, "bank", defs)
def_space = project_to_definition_space(embedded.embeddings, def_embs)
```

## Input Formats

| Format | Description |
|--------|-------------|
| **Paste sentences** | One sentence per line, grouped by corpus |
| **CSV / TSV** | Columns: `sentence`, `corpus` (required); `start`, `end` (optional positions) |
| **Pre-computed embeddings** | `.pt` or `.npy` file + metadata CSV (same row order) |

## Metrics

| Metric | Description |
|--------|-------------|
| **APD** | Average Pairwise Distance — global distributional divergence |
| **PRT** | Prototype Distance — cosine distance between corpus centroids |
| **AMD** | Average Minimum Distance — local correspondence between usages |
| **SAMD** | Symmetric AMD — greedy one-to-one matching between corpora |
| **AMD(C1→C2)** | Directional: high = senses in C1 not found in C2 (narrowing) |
| **AMD(C2→C1)** | Directional: high = senses in C2 not found in C1 (broadening) |

## Citation

If you use this tool in your research, please cite the underlying metrics paper:

```bibtex
@inproceedings{goworek-dubossarsky-2026-rethinking,
    title = "Rethinking Metrics for Lexical Semantic Change Detection",
    author = "Goworek, Roksana and Dubossarsky, Haim",
    booktitle = "Proceedings of the 6th International Workshop on Computational Approaches to Language Change (LChange'26)",
    year = "2026",
    pages = "147--161",
    url = "https://aclanthology.org/2026.lchange-1.13",
}
```

## Project Structure

```
semlens/
├── semlens/          # Core Python library (pip-installable)
│   ├── data_loading.py  # Input normalisation
│   ├── embeddings.py    # HF model loading & embedding extraction
│   ├── definitions.py   # Wiktionary, formatting, definition embedding
│   ├── spaces.py        # Full space & definition-space projection
│   ├── metrics.py       # APD, PRT, AMD, SAMD, directional AMD
│   ├── reduction.py     # PCA, UMAP, t-SNE for 2D visualisation
│   ├── lda.py           # LDA projection & weight extraction
│   └── utils.py         # Palettes, edit distance, text formatting
├── app/                 # Streamlit web application
│   ├── app.py           # Main entry point
│   ├── state.py         # Session state management
│   └── components/      # Reusable UI components
├── examples/            # Example data & definitions
└── tests/               # Unit tests
```

## Acknowledgements

This tool builds on code and methodology from the [Rethinking LSCD Metrics](https://github.com/roksanagow/Rethinking_LSCD_Metrics) repository and the [projecting_sentences](https://github.com/roksanagow/projecting_sentences) annotation tool.

## License

MIT
