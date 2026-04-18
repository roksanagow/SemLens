"""SemLens — Interactive Semantic Change Analysis.

Launch with:  streamlit run app/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semlens.data_loading import (
    TargetWordData,
    load_from_csv,
    load_from_sentences,
    load_from_raw_text,
    load_precomputed_embeddings,
)
from semlens.embeddings import (
    EmbeddedUsages,
    MODEL_ALIASES,
    embed_usages,
    embedded_usages_from_precomputed,
    load_model,
)
from semlens.definitions import (
    embed_definitions,
    fetch_wiktionary_definitions,
    format_definitions,
    load_definitions_from_json,
)
from semlens.lda import lda_definition_weights, lda_projection, lda_transform_new_points
from semlens.metrics import compute_all_metrics, compute_per_definition_metrics
from semlens.reduction import available_methods, reduce_to_2d
from semlens.spaces import project_to_definition_space
from semlens.utils import get_palette, palette_names

from app.state import init_state, reset_downstream_of
from app.annotation_store import (
    add_sense_class,
    clear_annotations,
    get_active_sense_class,
    get_annotations,
    get_sense_classes,
    remove_annotation,
    remove_sense_class,
    set_active_sense_class,
    set_annotation,
)
from app.components.scatter import render_lda_weights_chart, render_scatter


PROJECTION_HELP = (
    "PCA: linear, fast, global structure. "
    "UMAP: non-linear, preserves neighborhoods/clusters. "
    "t-SNE: non-linear, strongest local grouping, slower."
)

LDA_HELP = (
    "LDA view uses LD1 (best separating direction between corpora) and "
    "PC1 of residual variance for the second axis."
)

METRIC_HELP = {
    "apd": "APD: average cross-corpus pairwise distance. Higher means stronger shift.",
    "prt": "PRT: distance between corpus centroids (prototype shift).",
    "amd": "AMD: symmetric nearest-neighbor mismatch between corpora.",
    "samd": "SAMD: one-to-one greedy matching distance; robust to density imbalance.",
    "dir_1": "Directional AMD from first corpus to second; high suggests narrowing/disappearance.",
    "dir_2": "Directional AMD from second corpus to first; high suggests broadening/emergence.",
}

_USAGE_CAP_SAMPLE_SEED = 42


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{component:02x}" for component in rgb)


def _blend_with_white(hex_color: str, strength: float) -> str:
    strength = max(0.0, min(1.0, float(strength)))
    base_r, base_g, base_b = _hex_to_rgb(hex_color)
    red = int(round(255 * (1 - strength) + base_r * strength))
    green = int(round(255 * (1 - strength) + base_g * strength))
    blue = int(round(255 * (1 - strength) + base_b * strength))
    return _rgb_to_hex((red, green, blue))


def _column_gradient_styles(series: pd.Series, base_color: str) -> list[str]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return [""] * len(series)

    min_value = float(valid.min())
    max_value = float(valid.max())

    if min_value == max_value:
        strengths = pd.Series([0.75] * len(series), index=series.index)
    else:
        strengths = (values - min_value) / (max_value - min_value)

    styles: list[str] = []
    for strength in strengths:
        if pd.isna(strength):
            styles.append("")
        else:
            color = _blend_with_white(base_color, 0.18 + 0.82 * float(strength))
            styles.append(f"background-color: {color}; color: #111111;")
    return styles


def _cap_usages_by_corpus(
    data: TargetWordData,
    cap_by_corpus: dict[str, int],
) -> tuple[TargetWordData, list[int]]:
    """Cap usages per corpus via deterministic random sampling.

    A cap <= 0 means no cap for that corpus.
    Returns filtered data and kept original indices.
    """
    rng = np.random.default_rng(_USAGE_CAP_SAMPLE_SEED)
    kept_indices: list[int] = []

    for corpus, cap_value in cap_by_corpus.items():
        cap = int(cap_value)
        corpus_indices = data.indices_for_corpus(corpus)
        if cap <= 0 or cap >= len(corpus_indices):
            kept_indices.extend(corpus_indices)
            continue

        sampled = np.array(corpus_indices, dtype=int)
        rng.shuffle(sampled)
        kept_indices.extend(sorted(sampled[:cap].tolist()))

    if not cap_by_corpus:
        kept_indices = list(range(len(data.usages)))
    else:
        selected = set(kept_indices)
        for idx, usage in enumerate(data.usages):
            if usage.corpus not in cap_by_corpus and idx not in selected:
                kept_indices.append(idx)

    kept_indices = sorted(set(kept_indices))
    kept_usages = [data.usages[idx] for idx in kept_indices]
    return TargetWordData(word=data.word, usages=kept_usages), kept_indices


def _stable_annotation_coords(
    cache_key: str,
    n_points: int,
    compute_coords,
) -> np.ndarray:
    """Return cached annotation coordinates for deterministic selection behavior."""
    cache = st.session_state.get("_annot_coords_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["_annot_coords_cache"] = cache
    cached = cache.get(cache_key)

    if (
        cached is not None
        and isinstance(cached, np.ndarray)
        and cached.shape[0] == n_points
        and cached.shape[1] == 2
    ):
        return cached

    coords = np.asarray(compute_coords(), dtype=float)
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    cache[cache_key] = coords
    return coords


def _get_cached_model(model_id: str):
    """Reuse a previously loaded model if the model ID is unchanged."""
    cached_model_id = st.session_state.get("_loaded_model_id")
    cached_model = st.session_state.get("_loaded_model")
    if cached_model is not None and cached_model_id == model_id:
        return cached_model

    loaded = load_model(model_id, device="cpu")
    st.session_state["_loaded_model_id"] = model_id
    st.session_state["_loaded_model"] = loaded
    return loaded


def _shorten_label(label: str, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _render_square_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    *,
    colorscale: str = "RdBu",
    colorbar_title: str = "value",
    ignore_diagonal_for_scale: bool = False,
):
    """Render a square matrix heatmap with shared row/column labels."""
    display_labels = [_shorten_label(lbl) for lbl in labels]
    matrix_np = np.asarray(matrix, dtype=float)
    n = len(labels)
    hover_customdata = np.empty((n, n, 2), dtype=object)
    for row_i, y_label in enumerate(labels):
        for col_j, x_label in enumerate(labels):
            hover_customdata[row_i, col_j, 0] = x_label
            hover_customdata[row_i, col_j, 1] = y_label

    zmin = None
    zmax = None
    if ignore_diagonal_for_scale and n > 1:
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_values = matrix_np[off_diag_mask]
        off_diag_values = off_diag_values[np.isfinite(off_diag_values)]
        if off_diag_values.size > 0:
            min_val = float(np.min(off_diag_values))
            max_val = float(np.max(off_diag_values))
            if min_val < max_val:
                zmin = min_val
                zmax = max_val

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_np,
            x=display_labels,
            y=display_labels,
            customdata=hover_customdata,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "x: %{customdata[0]}"
                "<br>y: %{customdata[1]}"
                "<br>value: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=20, r=20, t=50, b=90),
        height=max(380, 36 * len(display_labels) + 160),
    )
    fig.update_xaxes(tickangle=45, side="bottom")
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SemLens",
    page_icon="🔍",
    layout="wide",
)

init_state()

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 SemLens")
    st.caption("Interpretable Semantic Change Analysis")
    st.divider()

    st.session_state.palette_name = st.selectbox(
        "Colour palette",
        palette_names(),
        index=palette_names().index(st.session_state.palette_name),
    )

    # Show palette preview
    pal = get_palette(st.session_state.palette_name)
    cols = st.columns(len(pal))
    for i, (col, c) in enumerate(zip(cols, pal)):
        col.markdown(
            f'<div style="background-color:{c};width:100%;height:18px;'
            f'border-radius:3px;"></div>',
            unsafe_allow_html=True,
        )

    st.session_state.reduction_method = st.selectbox(
        "2D projection method",
        available_methods(),
        index=0,
        help=PROJECTION_HELP,
    )

    st.session_state.distance_metric = st.selectbox(
        "Metric distance",
        ["cosine", "euclidean"],
        index=0 if st.session_state.distance_metric == "cosine" else 1,
        help="Used for APD, PRT, AMD, and SAMD in all metric panels.",
    )

    st.divider()

    # Status summary
    if st.session_state.target_word_data:
        data = st.session_state.target_word_data
        st.success(f"**Word:** {data.word}")
        st.write(f"**Usages:** {len(data)}")
        for c in data.corpus_labels:
            n = len(data.indices_for_corpus(c))
            st.write(f"  • *{c}*: {n}")
    if st.session_state.embedded_usages:
        st.success(f"**Embeddings:** {st.session_state.embedded_usages.dim}D")
    if st.session_state.definitions_formatted:
        st.success(f"**Definitions:** {len(st.session_state.definitions_formatted)}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_full, tab_defs, tab_annot = st.tabs([
    "📂 Data & Model",
    "🔵 Full Space",
    "📖 Definitions & Def Space",
    "🏷️ Annotation",
])


# =========================================================================
# TAB 1: Data & Model
# =========================================================================

with tab_data:
    st.header("Data Input")

    input_mode = st.radio(
        "Input format",
        ["Paste sentences", "Upload CSV / TSV", "Upload raw corpora (.txt)", "Upload pre-computed embeddings"],
        horizontal=True,
        help="Pick how to provide usages: sentence text, structured table, raw corpora, or pre-computed vectors.",
    )

    if input_mode == "Paste sentences":
        st.markdown(
            "Enter sentences grouped by corpus. Each line is one sentence. "
            "The tool will auto-detect the target word's position."
        )
        col_word, col_c1name, col_c2name = st.columns(3)
        word = col_word.text_input("Target word", value=st.session_state.target_word) or ""
        c1_name = col_c1name.text_input("Corpus 1 name", value="corpus_1")
        c2_name = col_c2name.text_input("Corpus 2 name", value="corpus_2")

        col1, col2 = st.columns(2)
        c1_text = col1.text_area(
            f"Sentences from **{c1_name}**",
            height=200,
            placeholder="One sentence per line...",
        )
        c2_text = col2.text_area(
            f"Sentences from **{c2_name}**",
            height=200,
            placeholder="One sentence per line...",
        )

        cap_col1, cap_col2 = st.columns(2)
        cap_c1 = int(cap_col1.number_input(
            f"Max usages from {c1_name}",
            min_value=0,
            value=0,
            step=1,
            key="paste_cap_c1",
            help="0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))
        cap_c2 = int(cap_col2.number_input(
            f"Max usages from {c2_name}",
            min_value=0,
            value=0,
            step=1,
            key="paste_cap_c2",
            help="0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))

        if st.button("Load data", type="primary", key="load_paste"):
            if not word.strip():
                st.error("Please enter a target word.")
            else:
                sents_1 = [s.strip() for s in c1_text.strip().split("\n") if s.strip()]
                sents_2 = [s.strip() for s in c2_text.strip().split("\n") if s.strip()]
                if not sents_1 or not sents_2:
                    st.error("Please enter sentences for both corpora.")
                else:
                    try:
                        all_sents = sents_1 + sents_2
                        all_labels = [c1_name] * len(sents_1) + [c2_name] * len(sents_2)
                        data = load_from_sentences(all_sents, word.strip(), all_labels)
                        data, _ = _cap_usages_by_corpus(data, {c1_name: cap_c1, c2_name: cap_c2})
                        reset_downstream_of("data")
                        st.session_state.target_word = word.strip()
                        st.session_state.target_word_data = data
                        st.session_state.corpus_labels = data.corpus_labels
                        st.success(f"Loaded {len(data)} usages of '{word}'.")
                    except Exception as e:
                        st.error(f"Error loading data: {e}")

    elif input_mode == "Upload CSV / TSV":
        st.markdown(
            "Upload a file with columns: **sentence**, **corpus** (required); "
            "**start**, **end** (optional character positions)."
        )
        uploaded = st.file_uploader(
            "Choose CSV or TSV",
            type=["csv", "tsv", "txt"],
            help="Required columns: sentence, corpus. Optional: start/end character offsets for the target word.",
        )
        word = st.text_input("Target word", value=st.session_state.target_word, key="csv_word") or ""
        cap_col1, cap_col2 = st.columns(2)
        cap_csv_1 = int(cap_col1.number_input(
            "Max usages from corpus 1",
            min_value=0,
            value=0,
            step=1,
            key="csv_cap_1",
            help="Applied to the first corpus label found in the file (sorted order). 0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))
        cap_csv_2 = int(cap_col2.number_input(
            "Max usages from corpus 2",
            min_value=0,
            value=0,
            step=1,
            key="csv_cap_2",
            help="Applied to the second corpus label found in the file (sorted order). 0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))

        if uploaded and st.button("Load data", type="primary", key="load_csv"):
            if not word.strip():
                st.error("Please enter a target word.")
            else:
                try:
                    # Save to temp file for the loader
                    suffix = ".tsv" if uploaded.name.endswith(".tsv") else ".csv"
                    tmp = Path(f"/tmp/semlens_upload{suffix}")
                    tmp.write_bytes(uploaded.read())
                    data = load_from_csv(tmp, word.strip())
                    labels = data.corpus_labels
                    cap_map = {}
                    if labels:
                        cap_map[labels[0]] = cap_csv_1
                    if len(labels) > 1:
                        cap_map[labels[1]] = cap_csv_2
                    data, _ = _cap_usages_by_corpus(data, cap_map)
                    reset_downstream_of("data")
                    st.session_state.target_word = word.strip()
                    st.session_state.target_word_data = data
                    st.session_state.corpus_labels = data.corpus_labels
                    st.success(f"Loaded {len(data)} usages of '{word}' from {uploaded.name}.")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

    elif input_mode == "Upload raw corpora (.txt)":
        st.markdown(
            "Upload two plain-text corpus files. The tool will split the text "
            "into sentences (on full stops) and extract only those containing "
            "an exact match of the target word."
        )
        word = st.text_input("Target word", value=st.session_state.target_word, key="corpus_word") or ""
        col_c1name, col_c2name = st.columns(2)
        c1_name = col_c1name.text_input("Corpus 1 name", value="corpus_1", key="corpus_c1name")
        c2_name = col_c2name.text_input("Corpus 2 name", value="corpus_2", key="corpus_c2name")
        col1, col2 = st.columns(2)
        c1_file = col1.file_uploader(
            f"**{c1_name}** (.txt)",
            type=["txt"],
            key="corpus1_upload",
            help="Plain text only. Sentences are extracted by full-stop splitting.",
        )
        c2_file = col2.file_uploader(
            f"**{c2_name}** (.txt)",
            type=["txt"],
            key="corpus2_upload",
            help="Plain text only. Sentences are extracted by full-stop splitting.",
        )
        cap_col1, cap_col2 = st.columns(2)
        cap_c1 = int(cap_col1.number_input(
            f"Max usages from {c1_name}",
            min_value=0,
            value=0,
            step=1,
            key="raw_cap_c1",
            help="0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))
        cap_c2 = int(cap_col2.number_input(
            f"Max usages from {c2_name}",
            min_value=0,
            value=0,
            step=1,
            key="raw_cap_c2",
            help="0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))

        if c1_file and c2_file and st.button("Load data", type="primary", key="load_corpus"):
            if not word.strip():
                st.error("Please enter a target word.")
            else:
                try:
                    c1_text = c1_file.read().decode("utf-8")
                    c2_text = c2_file.read().decode("utf-8")
                    data = load_from_raw_text(
                        {c1_name: c1_text, c2_name: c2_text},
                        word.strip(),
                    )
                    data, _ = _cap_usages_by_corpus(data, {c1_name: cap_c1, c2_name: cap_c2})
                    reset_downstream_of("data")
                    st.session_state.target_word = word.strip()
                    st.session_state.target_word_data = data
                    st.session_state.corpus_labels = data.corpus_labels
                    n1 = len(data.indices_for_corpus(c1_name))
                    n2 = len(data.indices_for_corpus(c2_name))
                    st.success(
                        f"Found {len(data)} sentences containing '{word}' "
                        f"({n1} from {c1_name}, {n2} from {c2_name})."
                    )
                except Exception as e:
                    st.error(f"Error loading corpora: {e}")

    elif input_mode == "Upload pre-computed embeddings":
        st.markdown(
            "Upload a `.pt` (PyTorch) or `.npy` (NumPy) file with shape `[N, D]`, "
            "**plus** a CSV/TSV with the sentence metadata (same row order)."
        )
        emb_file = st.file_uploader(
            "Embeddings file (.pt or .npy)",
            type=["pt", "npy"],
            help="Array/tensor shape must be [N, D]. N must match the number of metadata rows.",
        )
        meta_file = st.file_uploader(
            "Metadata CSV/TSV",
            type=["csv", "tsv", "txt"],
            key="meta_upload",
            help="Same row order as embeddings. Required columns: sentence, corpus.",
        )
        word = st.text_input("Target word", value=st.session_state.target_word, key="precomp_word") or ""
        cap_col1, cap_col2 = st.columns(2)
        cap_pre_1 = int(cap_col1.number_input(
            "Max usages from corpus 1",
            min_value=0,
            value=0,
            step=1,
            key="precomp_cap_1",
            help="Applied to the first corpus label found in metadata (sorted order). 0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))
        cap_pre_2 = int(cap_col2.number_input(
            "Max usages from corpus 2",
            min_value=0,
            value=0,
            step=1,
            key="precomp_cap_2",
            help="Applied to the second corpus label found in metadata (sorted order). 0 means no cap. Randomly samples from the provided sentences if cap is exceeded.",
        ))

        if emb_file and meta_file and st.button("Load data", type="primary", key="load_precomp"):
            if not word.strip():
                st.error("Please enter a target word.")
            else:
                try:
                    # Save embedding file
                    suffix = Path(emb_file.name).suffix
                    emb_path = Path(f"/tmp/semlens_embs{suffix}")
                    emb_path.write_bytes(emb_file.read())
                    embs = load_precomputed_embeddings(emb_path)

                    # Load metadata
                    meta_suffix = ".tsv" if meta_file.name.endswith(".tsv") else ".csv"
                    meta_path = Path(f"/tmp/semlens_meta{meta_suffix}")
                    meta_path.write_bytes(meta_file.read())
                    data = load_from_csv(meta_path, word.strip())
                    labels = data.corpus_labels
                    cap_map = {}
                    if labels:
                        cap_map[labels[0]] = cap_pre_1
                    if len(labels) > 1:
                        cap_map[labels[1]] = cap_pre_2
                    data, kept_indices = _cap_usages_by_corpus(data, cap_map)
                    embs = embs[kept_indices]

                    eu = embedded_usages_from_precomputed(data, embs)
                    reset_downstream_of("data")
                    st.session_state.target_word = word.strip()
                    st.session_state.target_word_data = data
                    st.session_state.corpus_labels = data.corpus_labels
                    st.session_state.embedded_usages = eu
                    st.session_state.embeddings_source = "precomputed"
                    st.success(
                        f"Loaded {len(data)} usages with pre-computed "
                        f"{embs.shape[1]}D embeddings."
                    )
                except Exception as e:
                    st.error(f"Error loading pre-computed embeddings: {e}")

    # --- Embedding section ---
    st.divider()
    st.header("Embedding Model")

    if st.session_state.embeddings_source == "precomputed" and st.session_state.embedded_usages:
        st.info("Using pre-computed embeddings. Skip this section.")
    else:
        alias_options = list(MODEL_ALIASES.keys())
        model_choice = st.selectbox(
            "Select a model (or type a HuggingFace model ID below)",
            ["(custom)"] + alias_options,
            index=0,
        )
        if model_choice == "(custom)":
            model_id = st.text_input(
                "HuggingFace model ID",
                value="pierluigic/xl-lexeme",
            )
        else:
            model_id = model_choice

        if st.button("Embed usages", type="primary", key="embed_btn"):
            data = st.session_state.target_word_data
            if data is None:
                st.error("Please load data first.")
            else:
                with st.spinner(
                    f"Embedding {len(data)} usages with `{model_id}` "
                    "(reuses loaded model when available)..."
                ):
                    try:
                        progress = st.progress(0)

                        def update_progress(batch, total):
                            progress.progress(batch / total)

                        loaded = _get_cached_model(model_id)
                        eu = embed_usages(
                            loaded, data,
                            progress_callback=update_progress,
                        )
                        reset_downstream_of("embeddings")
                        st.session_state.embedded_usages = eu
                        st.session_state.model_name = model_id
                        st.session_state.embeddings_source = "model"
                        progress.empty()
                        st.success(
                            f"Embedded {len(data)} usages → {eu.dim}D vectors."
                        )
                    except Exception as e:
                        st.error(f"Embedding error: {e}")


# =========================================================================
# TAB 2: Full Space
# =========================================================================

with tab_full:
    st.header("Full Embedding Space")

    eu = st.session_state.embedded_usages
    if eu is None:
        st.info("Load data and compute embeddings in the **Data & Model** tab first.")
    else:
        data = st.session_state.target_word_data
        labels = eu.corpus_labels
        assert len(labels) == 2, "Exactly 2 corpora required."
        c1, c2 = labels[0], labels[1]

        # --- View toggle ---
        view_mode = st.radio(
            "Visualisation",
            ["Dimensionality reduction", "LDA (LD1 + PC1)"],
            horizontal=True,
            key="full_view_mode",
            help=LDA_HELP,
        )

        # --- Compute metrics ---
        embs_c1 = eu.get_corpus_embeddings(c1)
        embs_c2 = eu.get_corpus_embeddings(c2)
        metrics = compute_all_metrics(
            embs_c1,
            embs_c2,
            (c1, c2),
            distance=st.session_state.distance_metric,
        )

        col_plot, col_metrics = st.columns([3, 1])

        with col_metrics:
            st.subheader("Metrics")
            st.markdown("*Computed on full-dimensional embeddings*")
            st.metric("APD", f"{metrics['apd']:.4f}", help=METRIC_HELP["apd"])
            st.metric("PRT", f"{metrics['prt']:.4f}", help=METRIC_HELP["prt"])
            st.metric("AMD", f"{metrics['amd']:.4f}", help=METRIC_HELP["amd"])
            st.metric("SAMD", f"{metrics['samd']:.4f}", help=METRIC_HELP["samd"])
            st.divider()
            st.markdown("**Directional AMD**")
            d_key1 = f"amd_{c1}_to_{c2}"
            d_key2 = f"amd_{c2}_to_{c1}"
            st.metric(f"{c1} → {c2}", f"{metrics[d_key1]:.4f}",
                       help=METRIC_HELP["dir_1"])
            st.metric(f"{c2} → {c1}", f"{metrics[d_key2]:.4f}",
                       help=METRIC_HELP["dir_2"])

        with col_plot:
            if view_mode == "Dimensionality reduction":
                coords = reduce_to_2d(
                    eu.embeddings,
                    method=st.session_state.reduction_method,
                )
                result = render_scatter(
                    coords,
                    data.corpus_list,
                    data.sentences,
                    data.word,
                    palette_name=st.session_state.palette_name,
                    title=f"Full space — {st.session_state.reduction_method.upper()}",
                    axis_labels=(
                        f"{st.session_state.reduction_method.upper()} 1",
                        f"{st.session_state.reduction_method.upper()} 2",
                    ),
                    key="full_scatter",
                )
            else:  # LDA
                lda_result = lda_projection(eu.embeddings, data.corpus_list)
                result = render_scatter(
                    lda_result.coords_2d,
                    data.corpus_list,
                    data.sentences,
                    data.word,
                    palette_name=st.session_state.palette_name,
                    title="Full space — LDA",
                    axis_labels=("LD1", "PC1 (residual)"),
                    key="full_lda_scatter",
                )


# =========================================================================
# TAB 3: Definitions & Definition Space
# =========================================================================

with tab_defs:
    st.header("Definitions")

    eu = st.session_state.embedded_usages
    if eu is None:
        st.info("Load data and compute embeddings in the **Data & Model** tab first.")
    else:
        data = st.session_state.target_word_data
        word = data.word
        labels = eu.corpus_labels
        c1, c2 = labels[0], labels[1]

        # --- Definition input ---
        def_source = st.radio(
            "Definition source",
            ["Manual input", "Fetch from Wiktionary", "Upload JSON"],
            horizontal=True,
        )

        if def_source == "Manual input":
            st.markdown("Enter one definition per line (without the word prefix — it's added automatically).")
            defs_text = st.text_area(
                "Definitions",
                value="\n".join(st.session_state.definitions_raw),
                height=200,
                placeholder="a financial institution\nthe side of a river\n...",
            )
            if st.button("Set definitions", type="primary", key="set_defs_manual"):
                raw = [d.strip() for d in defs_text.strip().split("\n") if d.strip()]
                if not raw:
                    st.error("Please enter at least one definition.")
                else:
                    reset_downstream_of("definitions")
                    st.session_state.definitions_raw = raw
                    st.session_state.definitions_formatted = format_definitions(word, raw)
                    st.success(f"Set {len(st.session_state.definitions_formatted)} definitions.")

        elif def_source == "Fetch from Wiktionary":
            wikt_lang = st.text_input("Wiktionary language code", value="en")
            if st.button("Fetch definitions", type="primary", key="fetch_wikt"):
                with st.spinner(f"Fetching definitions for '{word}' from {wikt_lang}.wiktionary.org..."):
                    try:
                        raw = fetch_wiktionary_definitions(word, language=wikt_lang)
                        if not raw:
                            st.warning("No definitions found. Try a different word or language.")
                        else:
                            reset_downstream_of("definitions")
                            st.session_state.definitions_raw = raw
                            st.session_state.definitions_formatted = format_definitions(word, raw)
                            st.success(f"Fetched {len(raw)} definitions.")
                    except Exception as e:
                        st.error(f"Wiktionary error: {e}")

        elif def_source == "Upload JSON":
            json_file = st.file_uploader("Definitions JSON", type=["json"])
            if json_file and st.button("Load definitions", type="primary", key="load_json_defs"):
                try:
                    tmp = Path("/tmp/semlens_defs.json")
                    tmp.write_bytes(json_file.read())
                    all_defs = load_definitions_from_json(tmp, word)
                    raw = all_defs[word]
                    reset_downstream_of("definitions")
                    st.session_state.definitions_raw = raw
                    st.session_state.definitions_formatted = format_definitions(word, raw)
                    st.success(f"Loaded {len(st.session_state.definitions_formatted)} definitions for '{word}'.")
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")

        # --- Show current definitions ---
        if st.session_state.definitions_formatted:
            with st.expander(f"Current definitions ({len(st.session_state.definitions_formatted)})", expanded=False):
                for i, d in enumerate(st.session_state.definitions_formatted):
                    st.write(f"{i+1}. {d}")

        # --- Embed definitions and project to def space ---
        if st.session_state.definitions_formatted and st.session_state.definition_embeddings is None:
            if st.button("Embed definitions & create definition space", type="primary", key="embed_defs"):
                with st.spinner("Embedding definitions..."):
                    try:
                        if st.session_state.embeddings_source == "precomputed":
                            st.error(
                                "Cannot embed definitions without a loaded model. "
                                "Please select a model in the Data & Model tab and "
                                "embed definitions using it."
                            )
                        else:
                            loaded = _get_cached_model(st.session_state.model_name)
                            def_embs = embed_definitions(
                                loaded, word,
                                st.session_state.definitions_formatted,
                            )
                            def_space = project_to_definition_space(eu.embeddings, def_embs)
                            st.session_state.definition_embeddings = def_embs
                            st.session_state.def_space_all = def_space
                            st.success(
                                f"Created definition space: "
                                f"{def_space.shape[1]} dimensions."
                            )
                    except Exception as e:
                        st.error(f"Error embedding definitions: {e}")

        # --- Definition Space Visualisation ---
        st.divider()
        st.header("Definition Space")

        def_space = st.session_state.def_space_all
        if def_space is None:
            if st.session_state.definitions_formatted:
                st.info("Click **Embed definitions & create definition space** above.")
            else:
                st.info("Add definitions above to create the definition space.")
        else:
            def_labels_clean = [
                d.split(": ", 1)[-1] if ": " in d else d
                for d in st.session_state.definitions_formatted
            ]

            st.subheader("Definition Similarity and Correlation")
            if len(def_labels_clean) < 2:
                st.info("Add at least two definitions to display pairwise heatmaps.")
            else:
                def_embs = st.session_state.definition_embeddings
                if def_embs is not None:
                    def_embs_np = def_embs.detach().cpu().numpy()
                    ignore_diag_for_scale = False
                    if st.session_state.distance_metric == "cosine":
                        norms = np.linalg.norm(def_embs_np, axis=1, keepdims=True)
                        norms = np.where(norms == 0.0, 1.0, norms)
                        normed = def_embs_np / norms
                        def_pairwise = np.clip(normed @ normed.T, -1.0, 1.0)
                        pairwise_title = "Definition-Definition Cosine Similarity"
                        pairwise_cbar = "cosine sim"
                        pairwise_scale = "RdBu"
                    else:
                        dists = np.linalg.norm(
                            def_embs_np[:, None, :] - def_embs_np[None, :, :],
                            axis=2,
                        )
                        # Convert Euclidean distance into a monotonic similarity score.
                        def_pairwise = 1.0 / (1.0 + dists)
                        pairwise_title = "Definition-Definition Euclidean Similarity"
                        pairwise_cbar = "euclid sim"
                        pairwise_scale = "RdBu"
                        ignore_diag_for_scale = True

                    with st.expander("Definition-Definition Similarity Heatmap", expanded=True):
                        _render_square_heatmap(
                            def_pairwise,
                            def_labels_clean,
                            pairwise_title,
                            colorscale=pairwise_scale,
                            colorbar_title=pairwise_cbar,
                            ignore_diagonal_for_scale=ignore_diag_for_scale,
                        )

                def_space_np = def_space.detach().cpu().numpy()
                corr = np.corrcoef(def_space_np, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                with st.expander("Definition-Dimension Correlation Heatmap", expanded=False):
                    _render_square_heatmap(
                        corr,
                        def_labels_clean,
                        "Correlation Between Definition Dimensions (Across Usages)",
                        colorscale="RdBu",
                        colorbar_title="pearson r",
                    )

            st.divider()

            view_mode_def = st.radio(
                "Visualisation",
                ["Dimensionality reduction", "LDA (LD1 + PC1)", "Definition axes (X/Y)"],
                horizontal=True,
                key="def_view_mode",
                help=LDA_HELP,
            )

            show_anchors = st.checkbox("Show definition anchors", value=False, key="show_def_anchors")

            # --- Definition highlight selector ---
            nearest_def_per_point = def_space.argmin(dim=1).tolist()

            highlight_def = st.selectbox(
                "Highlight usages closest to definition",
                ["(none)"] + def_labels_clean,
                index=0,
                key="highlight_def_select",
            )
            highlight_indices = None
            if highlight_def != "(none)":
                sel_def_idx = def_labels_clean.index(highlight_def)
                highlight_indices = [
                    i for i, nd in enumerate(nearest_def_per_point)
                    if nd == sel_def_idx
                ]
                if highlight_indices:
                    st.caption(f"{len(highlight_indices)} usage(s) closest to this definition")
                else:
                    st.caption("No usages are closest to this definition")

            # Compute definition space metrics
            idx_c1 = eu.get_corpus_indices(c1)
            idx_c2 = eu.get_corpus_indices(c2)
            def_c1 = def_space[idx_c1]
            def_c2 = def_space[idx_c2]
            def_metrics = compute_all_metrics(
                def_c1,
                def_c2,
                (c1, c2),
                distance=st.session_state.distance_metric,
            )

            col_plot_d, col_metrics_d = st.columns([3, 1])

            with col_metrics_d:
                st.subheader("Metrics (def space)")
                st.metric("APD", f"{def_metrics['apd']:.4f}", help=METRIC_HELP["apd"])
                st.metric("PRT", f"{def_metrics['prt']:.4f}", help=METRIC_HELP["prt"])
                st.metric("AMD", f"{def_metrics['amd']:.4f}", help=METRIC_HELP["amd"])
                st.metric("SAMD", f"{def_metrics['samd']:.4f}", help=METRIC_HELP["samd"])
                st.divider()
                st.markdown("**Directional AMD**")
                dk1 = f"amd_{c1}_to_{c2}"
                dk2 = f"amd_{c2}_to_{c1}"
                st.metric(f"{c1} → {c2}", f"{def_metrics[dk1]:.4f}", help=METRIC_HELP["dir_1"])
                st.metric(f"{c2} → {c1}", f"{def_metrics[dk2]:.4f}", help=METRIC_HELP["dir_2"])

            with col_plot_d:
                # Nearest definition per point (for hover)
                nearest_def_labels = [def_labels_clean[i] for i in nearest_def_per_point]

                # Definition anchor coordinates (if showing)
                anchor_coords = None
                anchor_labels = None
                if show_anchors and st.session_state.definition_embeddings is not None:
                    def_embs = st.session_state.definition_embeddings
                    anchor_in_def_space = project_to_definition_space(def_embs, def_embs)
                    combined = torch.cat([def_space, anchor_in_def_space], dim=0)
                else:
                    combined = def_space

                if view_mode_def == "Dimensionality reduction":
                    coords_all = reduce_to_2d(
                        combined,
                        method=st.session_state.reduction_method,
                    )
                    usage_coords = coords_all[:len(def_space)]

                    if show_anchors and st.session_state.definition_embeddings is not None:
                        anchor_coords = coords_all[len(def_space):]
                        anchor_labels = def_labels_clean

                    render_scatter(
                        usage_coords,
                        data.corpus_list,
                        data.sentences,
                        data.word,
                        palette_name=st.session_state.palette_name,
                        title=f"Definition space — {st.session_state.reduction_method.upper()}",
                        axis_labels=(
                            f"{st.session_state.reduction_method.upper()} 1",
                            f"{st.session_state.reduction_method.upper()} 2",
                        ),
                        definition_labels_per_point=nearest_def_labels,
                        definition_anchor_coords=anchor_coords,
                        definition_anchor_labels=anchor_labels,
                        show_definition_anchors=show_anchors,
                        highlight_indices=highlight_indices,
                        key="def_scatter",
                    )
                elif view_mode_def == "LDA (LD1 + PC1)":
                    lda_result = lda_projection(
                        def_space,
                        data.corpus_list,
                        feature_names=def_labels_clean,
                    )

                    # Project definition anchors through the same LDA transform
                    anchor_coords_lda = None
                    anchor_labels_lda = None
                    if show_anchors and st.session_state.definition_embeddings is not None:
                        def_embs = st.session_state.definition_embeddings
                        anchor_in_def_space = project_to_definition_space(def_embs, def_embs)
                        anchor_coords_lda = lda_transform_new_points(lda_result, anchor_in_def_space)
                        anchor_labels_lda = def_labels_clean

                    render_scatter(
                        lda_result.coords_2d,
                        data.corpus_list,
                        data.sentences,
                        data.word,
                        palette_name=st.session_state.palette_name,
                        title="Definition space — LDA",
                        axis_labels=("LD1", "PC1 (residual)"),
                        definition_labels_per_point=nearest_def_labels,
                        definition_anchor_coords=anchor_coords_lda,
                        definition_anchor_labels=anchor_labels_lda,
                        show_definition_anchors=show_anchors,
                        highlight_indices=highlight_indices,
                        key="def_lda_scatter",
                    )

                    # LDA weights bar chart
                    st.subheader("LDA Definition Weights")
                    weights_df = lda_definition_weights(lda_result, def_labels_clean)
                    render_lda_weights_chart(
                        weights_df,
                        palette_name=st.session_state.palette_name,
                        corpus_labels=(c1, c2),
                    )
                else:
                    col_x_def, col_y_def = st.columns(2)
                    x_def = col_x_def.selectbox(
                        "X-axis definition",
                        def_labels_clean,
                        index=0,
                        key="def_axis_x",
                    )
                    y_def = col_y_def.selectbox(
                        "Y-axis definition",
                        def_labels_clean,
                        index=1 if len(def_labels_clean) > 1 else 0,
                        key="def_axis_y",
                    )

                    x_idx = def_labels_clean.index(x_def)
                    y_idx = def_labels_clean.index(y_def)
                    usage_coords = def_space[:, [x_idx, y_idx]].detach().cpu().numpy()

                    anchor_coords_direct = None
                    anchor_labels_direct = None
                    if show_anchors and st.session_state.definition_embeddings is not None:
                        def_embs = st.session_state.definition_embeddings
                        anchor_in_def_space = project_to_definition_space(def_embs, def_embs)
                        anchor_coords_direct = anchor_in_def_space[:, [x_idx, y_idx]].detach().cpu().numpy()
                        anchor_labels_direct = def_labels_clean

                    render_scatter(
                        usage_coords,
                        data.corpus_list,
                        data.sentences,
                        data.word,
                        palette_name=st.session_state.palette_name,
                        title="Definition space — Direct axes",
                        axis_labels=(x_def, y_def),
                        definition_labels_per_point=nearest_def_labels,
                        definition_anchor_coords=anchor_coords_direct,
                        definition_anchor_labels=anchor_labels_direct,
                        show_definition_anchors=show_anchors,
                        highlight_indices=highlight_indices,
                        key="def_axes_scatter",
                    )

            # --- Per-definition metrics table ---
            st.divider()
            st.subheader("Per-Definition Metric Breakdown")
            st.markdown(
                "Each row shows LSCD metrics computed along a single definition "
                "dimension, sorted by the magnitude of the selected metric."
            )
            sort_col = st.selectbox(
                "Sort by",
                ["amd", "samd", f"amd_{c1}_to_{c2}", f"amd_{c2}_to_{c1}"],
                index=0,
                key="def_sort_by",
            )
            per_def_df = compute_per_definition_metrics(
                def_c1, def_c2, def_labels_clean,
                corpus_labels=(c1, c2),
                sort_by=sort_col,
                distance=st.session_state.distance_metric,
            )
            # Format for display
            display_df = per_def_df.drop(columns=["dim"], errors="ignore")
            numeric_cols = display_df.select_dtypes(include="number").columns
            styler = display_df.style.format({c: "{:.4f}" for c in numeric_cols})

            gradient_cmaps = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#17a2b8", "#e377c2"]
            for i, col_name in enumerate(numeric_cols):
                styler = styler.apply(
                    lambda col, color=gradient_cmaps[i % len(gradient_cmaps)]: _column_gradient_styles(col, color),
                    axis=0,
                    subset=[col_name],
                )

            st.dataframe(
                styler,
                width='stretch',
                height=min(400, 35 * len(display_df) + 40),
            )


# =========================================================================
# TAB 4: Annotation
# =========================================================================

with tab_annot:
    st.header("Sense Annotation")

    st.markdown(
        """
        <style>
        /* Make selected class buttons (secondary variant) look clearly lit. */
        div[data-testid="stButton"] > button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"] {
            background-color: #ffffff !important;
            color: #111111 !important;
            border: 1px solid #d6d6d6 !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #f7f7f7 !important;
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    eu = st.session_state.embedded_usages
    if eu is None:
        st.info("Load data and compute embeddings first.")
    else:
        data = st.session_state.target_word_data
        annotations = get_annotations()
        sense_classes = get_sense_classes()
        active_class = get_active_sense_class()

        if sense_classes and active_class not in sense_classes:
            repaired_class = sense_classes[0]
            set_active_sense_class(repaired_class)
            active_class = repaired_class

        # --- Class management (top bar) ---
        st.markdown(
            "**Workflow:** create sense classes below, select one as active, "
            "then click or lasso-select points in the plot to assign them. "
            "Click an already-assigned point to un-assign it."
        )

        col_new, col_active = st.columns([1, 1])

        with col_new:
            with st.form("add_class_form", clear_on_submit=False):
                new_col1, new_col2 = st.columns([3, 1])
                new_class = new_col1.text_input(
                    "New sense class name",
                    key="new_class_input",
                    label_visibility="collapsed",
                    placeholder="Type a new sense class name...",
                )
                submitted = new_col2.form_submit_button(
                    "Add class",
                    width='stretch',
                    type="tertiary",
                )
            if submitted:
                if new_class.strip() and new_class.strip() not in sense_classes:
                    created_class = new_class.strip()
                    add_sense_class(created_class)
                    set_active_sense_class(created_class)
                    st.rerun()

        with col_active:
            if sense_classes:
                selected_class = st.selectbox(
                    "Active annotation class",
                    sense_classes,
                    index=sense_classes.index(active_class)
                        if active_class in sense_classes else 0,
                )
                set_active_sense_class(selected_class)
                active_class = selected_class
            else:
                st.caption("Create a sense class to start annotating.")

        # --- Show classes with counts and delete buttons ---
        if sense_classes:
            sense_palette = get_palette("Tol Bright (colourblind)")
            class_cols = st.columns(min(len(sense_classes), 5))
            for i, sc in enumerate(sense_classes):
                count = sum(1 for v in annotations.values() if v == sc)
                colour = sense_palette[i % len(sense_palette)]
                is_active = sc == active_class
                with class_cols[i % len(class_cols)]:
                    dot_col, btn_col = st.columns([1, 6])
                    dot_col.markdown(
                        f'<div style="text-align:center; color:{colour}; font-size:18px; line-height:30px;">⬤</div>',
                        unsafe_allow_html=True,
                    )
                    if btn_col.button(
                        f"{sc} ({count})",
                        key=f"activate_{sc}",
                        width='stretch',
                        type="secondary" if is_active else "tertiary",
                    ):
                        set_active_sense_class(sc)
                        st.rerun()
                    if st.button("Remove", key=f"rm_{sc}", width='stretch', type="tertiary"):
                        remove_sense_class(sc)
                        st.rerun()

            n_unassigned = len(data) - len(annotations)
            st.caption(f"Unassigned: {n_unassigned} / {len(data)}")

        col_clear_btn, col_clear_msg = st.columns([1, 3])
        with col_clear_btn:
            if st.button("Delete all annotations", key="clear_all_annots_btn", width='stretch', type="tertiary"):
                st.session_state.confirm_clear_annotations = True
        with col_clear_msg:
            if st.session_state.get("confirm_clear_annotations", False):
                st.warning("Delete all annotations? This cannot be undone in the current session.")
                col_confirm, col_cancel = st.columns(2)
                if col_confirm.button("Confirm delete", key="confirm_clear_annots", width='stretch'):
                    clear_annotations()
                    st.session_state.confirm_clear_annotations = False
                    st.rerun()
                if col_cancel.button("Cancel", key="cancel_clear_annots", width='stretch'):
                    st.session_state.confirm_clear_annotations = False
                    st.rerun()

        st.divider()

        # --- View switching ---
        has_def_space = st.session_state.def_space_all is not None
        def_labels_clean = [
            d.split(": ", 1)[-1] if ": " in d else d
            for d in st.session_state.definitions_formatted
        ]

        annot_view_options = ["Full space — dim. reduction", "Full space — LDA"]
        if has_def_space:
            annot_view_options += [
                "Definition space — dim. reduction",
                "Definition space — LDA",
                "Definition space — definition axes",
            ]

        annot_view = st.radio(
            "Annotation view",
            annot_view_options,
            horizontal=True,
            key="annot_view_mode",
        )

        annot_interaction_mode = st.radio(
            "Annotation interaction",
            ["Click", "Lasso"],
            horizontal=True,
            key="annot_interaction_mode",
            help="Click: single-point assignment. Lasso: free-form multi-point assignment.",
        )

        # --- Compute coordinates for the selected view ---
        if annot_view == "Full space — dim. reduction":
            coords = _stable_annotation_coords(
                cache_key=(
                    f"full_reduced::{st.session_state.reduction_method}::"
                    f"{data.word}::{len(data)}"
                ),
                n_points=len(data),
                compute_coords=lambda: reduce_to_2d(
                    eu.embeddings,
                    method=st.session_state.reduction_method,
                ),
            )
            ax_labels = (
                f"{st.session_state.reduction_method.upper()} 1",
                f"{st.session_state.reduction_method.upper()} 2",
            )
            view_title = f"Annotation — Full space {st.session_state.reduction_method.upper()}"
            chart_key_suffix = f"full_reduced_{st.session_state.reduction_method}"
        elif annot_view == "Full space — LDA":
            coords = _stable_annotation_coords(
                cache_key=f"full_lda::{data.word}::{len(data)}",
                n_points=len(data),
                compute_coords=lambda: lda_projection(eu.embeddings, data.corpus_list).coords_2d,
            )
            ax_labels = ("LD1", "PC1 (residual)")
            view_title = "Annotation — Full space LDA"
            chart_key_suffix = "full_lda"
        elif annot_view == "Definition space — dim. reduction":
            coords = _stable_annotation_coords(
                cache_key=(
                    f"def_reduced::{st.session_state.reduction_method}::"
                    f"{data.word}::{len(data)}"
                ),
                n_points=len(data),
                compute_coords=lambda: reduce_to_2d(
                    st.session_state.def_space_all,
                    method=st.session_state.reduction_method,
                ),
            )
            ax_labels = (
                f"{st.session_state.reduction_method.upper()} 1",
                f"{st.session_state.reduction_method.upper()} 2",
            )
            view_title = f"Annotation — Def space {st.session_state.reduction_method.upper()}"
            chart_key_suffix = f"def_reduced_{st.session_state.reduction_method}"
        elif annot_view == "Definition space — LDA":
            coords = _stable_annotation_coords(
                cache_key=f"def_lda::{data.word}::{len(data)}",
                n_points=len(data),
                compute_coords=lambda: lda_projection(
                    st.session_state.def_space_all,
                    data.corpus_list,
                ).coords_2d,
            )
            ax_labels = ("LD1", "PC1 (residual)")
            view_title = "Annotation — Def space LDA"
            chart_key_suffix = "def_lda"
        elif annot_view == "Definition space — definition axes":
            col_x_def, col_y_def = st.columns(2)
            x_def = col_x_def.selectbox(
                "X-axis definition",
                def_labels_clean,
                index=0,
                key="annot_axis_x",
            )
            y_def = col_y_def.selectbox(
                "Y-axis definition",
                def_labels_clean,
                index=1 if len(def_labels_clean) > 1 else 0,
                key="annot_axis_y",
            )
            x_idx = def_labels_clean.index(x_def)
            y_idx = def_labels_clean.index(y_def)
            coords = _stable_annotation_coords(
                cache_key=f"def_axes::{x_idx}:{y_idx}::{data.word}::{len(data)}",
                n_points=len(data),
                compute_coords=lambda: st.session_state.def_space_all[:, [x_idx, y_idx]].detach().cpu().numpy(),
            )
            ax_labels = (x_def, y_def)
            view_title = "Annotation — Def space direct axes"
            chart_key_suffix = f"def_axes_{x_idx}_{y_idx}"
        else:
            coords = reduce_to_2d(eu.embeddings, method="pca")
            ax_labels = ("PC1", "PC2")
            view_title = "Annotation"
            chart_key_suffix = "fallback"

        chart_key = f"annot_chart_{chart_key_suffix}"

        # --- Scatter plot ---
        result = render_scatter(
            coords,
            data.corpus_list,
            data.sentences,
            data.word,
            annotations=annotations,
            sense_classes=sense_classes,
            palette_name=st.session_state.palette_name,
            title=view_title,
            axis_labels=ax_labels,
            enable_selection=True,
            interaction_mode=annot_interaction_mode.lower(),
            key=chart_key,
        )

        if annot_interaction_mode == "Click":
            st.caption("Click a point to assign/unassign it to the active class.")
        else:
            st.caption("Use the **lasso** tool to select multiple points and assign/unassign them at once.")

        # --- Handle selection: assign clicked points to active class ---
        selected_indices = result.get("clicked_points", [])
        active_class = get_active_sense_class()

        if selected_indices and active_class:
            changed = False
            for idx in selected_indices:
                if annotations.get(idx) == active_class:
                    # Toggle off: un-assign if already this class
                    remove_annotation(idx)
                    changed = True
                else:
                    # Assign to active class
                    set_annotation(idx, active_class)
                    changed = True
            if changed:
                st.rerun()
        elif selected_indices and not active_class:
            st.warning("Select an active sense class before clicking points.")

        # --- Annotated sentences list ---
        if annotations:
            st.divider()
            with st.expander("Annotated sentences", expanded=False):
                for sc in sense_classes:
                    sc_indices = sorted(k for k, v in annotations.items() if v == sc)
                    if not sc_indices:
                        continue
                    st.markdown(f"**{sc}** ({len(sc_indices)})")
                    for idx in sc_indices:
                        u = data.usages[idx]
                        col_sent, col_rm = st.columns([5, 1])
                        col_sent.caption(f"[{u.corpus}] {u.sentence}")
                        if col_rm.button("✕", key=f"rm_annot_{idx}"):
                            remove_annotation(idx)
                            st.rerun()

        # --- Export ---
        st.divider()
        st.subheader("Export Annotations")

        if annotations:
            rows = []
            for idx, sense in sorted(annotations.items()):
                u = data.usages[idx]
                rows.append({
                    "lemma": u.word,
                    "sentence": u.sentence,
                    "sense": sense,
                    "corpus": u.corpus,
                    "start": u.char_start,
                    "end": u.char_end,
                })
            export_df = pd.DataFrame(rows)

            st.dataframe(export_df, width='stretch', height=200)

            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{data.word}_annotated.csv",
                mime="text/csv",
            )
        else:
            st.caption("No annotations yet.")
