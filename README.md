# SemLens 

**Interactive lexical semantic change detection for comparing words across time, domains, or corpora.**

SemLens helps you inspect how the meaning of a target word shifts between two corpora. It combines contextual embeddings, dictionary definitions, interpretable projections, and semantic change metrics so you can move from a raw distributional comparison to a sense-level explanation.

Use it for:

- **Diachronic semantic change**: compare earlier vs. later language
- **Cross-domain shift**: compare the same word across genres, registers, or subject domains
- **Corpus comparison**: compare any two datasets where the same word appears in different contexts

## What SemLens Does

- Computes common **lexical semantic change detection (LSCD)** metrics: APD, PRT, AMD, SAMD, and directional AMD
- Lets you choose a global **metric distance** (cosine or Euclidean) for APD, PRT, AMD, and SAMD
- Visualises usages in a 2D scatter plot using PCA, UMAP, t-SNE, or LDA
- Builds a **definition-aligned space** so dictionary senses become explicit axes
- Lets you annotate usages into sense classes and export them
- Supports manual definitions, Wiktionary definitions, or JSON input

## Why Definition Projection Matters

The most intuitive embedding plot is not always the most useful one. In a normal 2D reduction, the axes are mathematical constructs and can be hard to interpret. SemLens also offers **definition projection**, which turns each dictionary definition into a semantic axis.

In practice, this means:

- each usage is scored by how close it is to each definition
- each dimension corresponds to one dictionary sense or gloss
- you can inspect which sense axes separate your corpora most strongly

So instead of asking only “do these corpora differ?”, you can ask “which sense is expanding, shrinking, or disappearing?”

## Main Features

- **LSCD metrics**: APD, PRT, AMD, SAMD, and directional AMD for asymmetric change detection
- **Distance control**: choose cosine or Euclidean distance globally from the sidebar for metric computation
- **Definition-space projection**: project usages onto dictionary definitions for an interpretable sense space
- **Interactive 2D plots**: hover to inspect sentences, colour by corpus, and switch between PCA / UMAP / t-SNE
- **LDA views**: separate corpora with LD1 and inspect interpretable definition weights
- **Per-definition metrics**: see which senses contribute most to the change
- **Definition diagnostics heatmaps**: inspect definition-definition similarity and cross-definition correlation in collapsible panels
- **Sense annotation**: create sense classes, annotate with click or lasso, and export annotations
- **Wiktionary support**: fetch definitions automatically via the MediaWiki API
- **Flexible input**: paste sentences, upload CSV/TSV, upload raw text, or load pre-computed embeddings
- **Accessible palettes**: colourblind-friendly palette options

## Installation

```bash
git clone https://github.com/roksanagow/semlens.git
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

### 1. Launch the web app

```bash
source .venv/bin/activate
streamlit run app/app.py
```

Open the URL shown in the terminal, usually `http://localhost:8501`.

### 2. Load your data

In the **Data & Model** tab, choose one of the following:

- paste sentences directly
- upload a CSV/TSV file
- upload raw text files
- load pre-computed embeddings plus metadata

### 3. Embed usages

Choose a model or enter a Hugging Face model ID, then embed the usages.

### 4. Add definitions

In the **Definitions & Def Space** tab, either:

- type definitions manually
- fetch them from Wiktionary
- load them from JSON

### 5. Explore the change

- inspect the full embedding space
- compare LSCD metrics
- switch to definition space to see which senses explain the change
- annotate usages into sense classes and export the results

### 6. Export annotations

In the **Annotation** tab, create sense classes, assign usages with click or lasso selection, and download the annotated CSV.

### Annotation Tips

- Choose an **Annotation interaction** mode:
    - **Click** for single-point annotation
    - **Lasso** for free-form multi-point annotation
- In click mode, selection is forgiving: you can click near a point (within the hover area), not only the exact marker center.
- Click a class button to make it active.
- Press **Enter** in the class-name box to add a new class quickly.
- New classes become active automatically when created.

## How Definition Projection Works

Definition projection is the key idea behind SemLens.

If a word has definitions such as:

- a financial institution
- the side of a river

then every usage is compared against both definitions. The result is a new space where each axis corresponds to one definition. A point near the “financial institution” axis is semantically closer to that sense; a point near the “river bank” axis is closer to that sense.

This makes it easier to understand *what kind* of change is happening, not just *whether* change is happening.

### Interpreting LDA Definition Weights in Definition Space

Definition-space dimensions are **distances** to each definition (cosine distance by construction):

- larger value = farther from that definition (less associated)
- smaller value = closer to that definition (more associated)

So LDA definition weights are currently weights over **distance features**.
If a definition has a positive LD1 weight, larger distance to that definition pushes a usage toward the second corpus side of LD1.
If it has a negative LD1 weight, larger distance pushes toward the first corpus side.

Practical reading tip: combine **weight sign** with each corpus's average definition distance in the per-definition table to decide which corpus is actually closer to that sense.

## Typical Workflow

1. Load two corpora you want to compare.
2. Embed the target word usages.
3. Compute LSCD metrics.
4. Add definitions for the target word.
5. Inspect the definition-aligned plot.
6. Annotate the usages into sense classes.
7. Export your annotations if you want to use them elsewhere.

## Programmatic Use

SemLens can also be used as a Python library.

```python
from semlens.data_loading import load_from_sentences
from semlens.embeddings import load_model, embed_usages
from semlens.metrics import compute_all_metrics
from semlens.definitions import format_definitions, embed_definitions
from semlens.spaces import project_to_definition_space

# Load usages
data = load_from_sentences(
    sentences=["I deposited money at the bank", "The river bank was muddy"],
    word="bank",
    corpus_labels=["modern", "modern"],
)

# Embed usages
model = load_model("xl-lexeme")
embedded = embed_usages(model, data)

# Compare corpora
metrics = compute_all_metrics(
    embedded.get_corpus_embeddings("modern"),
    embedded.get_corpus_embeddings("modern"),
)

# Build definition space
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
| **Raw text files** | One plain-text file per corpus; sentences are extracted automatically |
| **Pre-computed embeddings** | `.pt` or `.npy` file + metadata CSV/TSV (same row order) |

## Metrics

| Metric | Description |
|--------|-------------|
| **APD** | Average Pairwise Distance — global distributional divergence |
| **PRT** | Prototype Distance — distance between corpus centroids |
| **AMD** | Average Minimum Distance — local correspondence between usages |
| **SAMD** | Symmetric AMD — greedy one-to-one matching between corpora |
| **AMD(C1→C2)** | Directional: high = senses in C1 not found in C2 (narrowing) |
| **AMD(C2→C1)** | Directional: high = senses in C2 not found in C1 (broadening) |

All metric panels use the sidebar **Metric distance** selector:

- cosine
- euclidean

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
