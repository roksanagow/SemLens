"""Plotly scatter plot component with hover, click-to-annotate, and corpus colouring."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from semlens.utils import (
    bold_word_in_sentence,
    format_hover_text,
    get_palette,
    split_text,
)


def _points_in_polygon(points: np.ndarray, poly_x: list[float], poly_y: list[float]) -> list[int]:
    """Return indices of points inside a polygon via ray casting."""
    if len(poly_x) < 3 or len(poly_y) < 3:
        return []

    xs = np.asarray(poly_x, dtype=float)
    ys = np.asarray(poly_y, dtype=float)
    xq = points[:, 0].astype(float)
    yq = points[:, 1].astype(float)

    # Ensure closed polygon for edge iteration
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])

    inside = np.zeros(points.shape[0], dtype=bool)
    j = len(xs) - 1
    for i in range(len(xs)):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]

        intersects = ((yi > yq) != (yj > yq)) & (
            xq < (xj - xi) * (yq - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= intersects
        j = i

    return np.where(inside)[0].tolist()


def _extract_lasso_xy(selection: object) -> tuple[list[float], list[float]]:
    """Extract lasso polygon x/y arrays from selection payload variants."""
    if not hasattr(selection, "get"):
        return [], []

    selection_obj: Any = selection
    lasso = selection_obj.get("lasso", {})
    if not hasattr(lasso, "get"):
        return [], []

    x = lasso.get("x", [])
    y = lasso.get("y", [])
    if isinstance(x, list) and isinstance(y, list):
        return x, y

    # Some payload variants encode the lasso as an SVG-like path string.
    path = lasso.get("path", "")
    if isinstance(path, str) and path:
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", path)]
        if len(nums) >= 6 and len(nums) % 2 == 0:
            px = nums[0::2]
            py = nums[1::2]
            return px, py

    return [], []


def _extract_box_bounds(selection: object) -> tuple[float, float, float, float] | None:
    """Extract box-select bounds from selection payload variants."""
    if not hasattr(selection, "get"):
        return None

    selection_obj: Any = selection
    box = selection_obj.get("box", {})
    if not hasattr(box, "get"):
        return None

    x = box.get("x", [])
    y = box.get("y", [])
    if isinstance(x, list) and isinstance(y, list) and x and y:
        return float(min(x)), float(max(x)), float(min(y)), float(max(y))

    if all(k in box for k in ("x0", "x1", "y0", "y1")):
        try:
            return (
                float(min(box["x0"], box["x1"])),
                float(max(box["x0"], box["x1"])),
                float(min(box["y0"], box["y1"])),
                float(max(box["y0"], box["y1"])),
            )
        except (TypeError, ValueError):
            return None

    return None


def _build_figure_multi_trace(
    coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
    annotations, sense_classes, definition_anchor_coords, definition_anchor_labels,
    show_definition_anchors, highlight_indices,
):
    """Build figure with one trace per corpus (for display-only charts)."""
    fig = go.Figure()

    for corpus_name in unique_corpora:
        mask = [i for i, c in enumerate(corpus_per_point) if c == corpus_name]
        colour = corpus_colour_map[corpus_name]

        marker_line_widths = []
        marker_line_colours = []
        for idx in mask:
            if annotations and idx in annotations:
                marker_line_widths.append(2.5)
                marker_line_colours.append("#333333")
            else:
                marker_line_widths.append(0)
                marker_line_colours.append(colour)

        fig.add_trace(go.Scatter(
            x=coords_2d[mask, 0].tolist(),
            y=coords_2d[mask, 1].tolist(),
            mode="markers",
            name=corpus_name,
            text=[hover_texts[i] for i in mask],
            hoverinfo="text",
            marker=dict(
                size=9, color=colour, opacity=0.8,
                line=dict(width=marker_line_widths, color=marker_line_colours),
            ),
        ))

    # Annotation overlay
    if annotations and sense_classes:
        sense_palette = get_palette("Tol Bright (colourblind)")
        sense_colour_map = {s: sense_palette[i % len(sense_palette)] for i, s in enumerate(sense_classes)}
        for sense_name in sense_classes:
            sense_idx = [i for i, s in annotations.items() if s == sense_name]
            if not sense_idx:
                continue
            fig.add_trace(go.Scatter(
                x=coords_2d[sense_idx, 0].tolist(),
                y=coords_2d[sense_idx, 1].tolist(),
                mode="markers",
                name=f"⬡ {sense_name}",
                text=[hover_texts[i] for i in sense_idx],
                hoverinfo="text",
                marker=dict(
                    size=13, color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color=sense_colour_map[sense_name]),
                    symbol="diamond",
                ),
            ))

    # Definition anchors
    if show_definition_anchors and definition_anchor_coords is not None and definition_anchor_labels is not None:
        fig.add_trace(go.Scatter(
            x=definition_anchor_coords[:, 0].tolist(),
            y=definition_anchor_coords[:, 1].tolist(),
            mode="markers+text",
            name="Definitions",
            text=definition_anchor_labels,
            textposition="top center",
            textfont=dict(size=9, color="#555555"),
            hoverinfo="text",
            marker=dict(size=12, color="#999999", symbol="star", line=dict(width=1, color="#333333")),
        ))

    # Highlight
    if highlight_indices is not None and len(highlight_indices) > 0:
        fig.add_trace(go.Scatter(
            x=coords_2d[highlight_indices, 0].tolist(),
            y=coords_2d[highlight_indices, 1].tolist(),
            mode="markers",
            name="Highlighted",
            text=[hover_texts[i] for i in highlight_indices],
            hoverinfo="text",
            marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(width=3, color="#FF0000"), symbol="circle"),
            showlegend=True,
        ))

    return fig


def _build_figure_single_trace(
    coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
    annotations, sense_classes, interaction_mode,
):
    """Build figure with ONE trace containing all points (for annotation charts).

    Using a single trace avoids Plotly/Streamlit issues with lasso selection
    across multiple traces.  Corpus membership is shown via per-point colours.
    """
    fig = go.Figure()

    n = len(corpus_per_point)
    all_indices = list(range(n))

    # In click mode, add a larger transparent hit area so users can select
    # points without needing pixel-perfect clicks on tiny markers.
    if interaction_mode == "click":
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0].tolist(),
            y=coords_2d[:, 1].tolist(),
            mode="markers",
            name="_hit_area",
            text=hover_texts,
            hoverinfo="text",
            customdata=all_indices,
            marker=dict(
                size=22,
                color="rgba(0,0,0,0.001)",
                opacity=1.0,
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            showlegend=False,
        ))

    # Per-point colours based on corpus
    colours = [corpus_colour_map[c] for c in corpus_per_point]

    # Annotation outlines
    marker_line_widths = [0.0] * n
    marker_line_colours = ["rgba(0,0,0,0)"] * n
    if annotations:
        for idx in annotations:
            if 0 <= idx < n:
                marker_line_widths[idx] = 2.5
                marker_line_colours[idx] = "#333333"

    # Main trace — all points, customdata = original index
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0].tolist(),
        y=coords_2d[:, 1].tolist(),
        mode="markers",
        name="usages",
        text=hover_texts,
        hoverinfo="text",
        customdata=all_indices,
        marker=dict(
            size=9,
            color=colours,
            opacity=0.8,
            line=dict(width=marker_line_widths, color=marker_line_colours),
        ),
        showlegend=False,
    ))

    # Invisible legend traces for corpus colours
    for corpus_name in unique_corpora:
        colour = corpus_colour_map[corpus_name]
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=corpus_name,
            marker=dict(size=9, color=colour),
            showlegend=True,
        ))

    # Annotation overlay — sense diamonds
    if annotations and sense_classes:
        sense_palette = get_palette("Tol Bright (colourblind)")
        sense_colour_map = {s: sense_palette[i % len(sense_palette)] for i, s in enumerate(sense_classes)}
        for sense_name in sense_classes:
            sense_idx = [i for i, s in annotations.items() if s == sense_name]
            if not sense_idx:
                continue
            fig.add_trace(go.Scatter(
                x=coords_2d[sense_idx, 0].tolist(),
                y=coords_2d[sense_idx, 1].tolist(),
                mode="markers",
                name=f"⬡ {sense_name}",
                text=[hover_texts[i] for i in sense_idx],
                hoverinfo="text",
                customdata=sense_idx,
                marker=dict(
                    size=13, color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color=sense_colour_map[sense_name]),
                    symbol="diamond",
                ),
            ))

    return fig


def render_scatter(
    coords_2d: np.ndarray,
    corpus_per_point: list[str],
    sentences: list[str],
    target_word: str,
    *,
    annotations: dict[int, str] | None = None,
    sense_classes: list[str] | None = None,
    palette_name: str = "Tab10 (matplotlib)",
    title: str = "Usage Embeddings",
    axis_labels: tuple[str, str] = ("Dim 1", "Dim 2"),
    definition_labels_per_point: list[str] | None = None,
    definition_anchor_coords: np.ndarray | None = None,
    definition_anchor_labels: list[str] | None = None,
    show_definition_anchors: bool = False,
    highlight_indices: list[int] | None = None,
    enable_selection: bool = False,
    interaction_mode: str = "lasso",
    key: str = "scatter",
) -> dict:
    """Render an interactive Plotly scatter plot in Streamlit.

    Parameters
    ----------
    enable_selection : bool
        If True, uses a single-trace figure with lasso/box selection enabled.
        Only set for the annotation chart.

    Returns
    -------
    dict
        ``{"clicked_points": list[int]}`` — selected point indices.
    """
    palette = get_palette(palette_name)
    unique_corpora = sorted(set(corpus_per_point))
    corpus_colour_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_corpora)}

    hover_texts = []
    for i, (sent, corp) in enumerate(zip(sentences, corpus_per_point)):
        nearest_def = definition_labels_per_point[i] if definition_labels_per_point else None
        hover_texts.append(format_hover_text(sent, target_word, corp, nearest_definition=nearest_def))

    if enable_selection:
        fig = _build_figure_single_trace(
            coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
            annotations, sense_classes, interaction_mode,
        )
    else:
        fig = _build_figure_multi_trace(
            coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
            annotations, sense_classes, definition_anchor_coords, definition_anchor_labels,
            show_definition_anchors, highlight_indices,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        hovermode="closest",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    if enable_selection:
        if interaction_mode == "click":
            fig.update_layout(dragmode="pan", clickmode="event+select")
        else:
            fig.update_layout(dragmode="lasso", clickmode="event+select")

    # --- Render ---
    if not enable_selection:
        st.plotly_chart(fig, width='stretch', key=key)
        return {"clicked_points": []}

    event = st.plotly_chart(
        fig,
        width='stretch',
        key=key,
        on_select="rerun",
        selection_mode=("points",) if interaction_mode == "click" else ("lasso",),
    )

    # --- Parse selection ---
    clicked_points = []
    try:
        selection = event.get("selection", {})
        points = selection.get("points", []) if isinstance(selection, dict) else []

        lasso_x, lasso_y = _extract_lasso_xy(selection)
        box_bounds = _extract_box_bounds(selection)

        for pt in points:
            cd = pt.get("customdata")
            if cd is None:
                # Some payload variants include point index fields only.
                pi = pt.get("pointIndex", pt.get("pointNumber"))
                if isinstance(pi, (int, np.integer)):
                    clicked_points.append(int(pi))
                continue
            if isinstance(cd, (list, tuple)):
                for item in cd:
                    if isinstance(item, (int, float, np.integer, np.floating)):
                        clicked_points.append(int(item))
            elif isinstance(cd, (int, float, np.integer, np.floating)):
                clicked_points.append(int(cd))

        # Alternative selection payload used by some Plotly/Streamlit versions.
        if isinstance(selection, dict):
            point_indices = selection.get("point_indices", [])
            if isinstance(point_indices, list):
                for idx in point_indices:
                    if isinstance(idx, (int, np.integer)):
                        clicked_points.append(int(idx))

        if hasattr(event, "get"):
            event_point_indices = event.get("point_indices", [])
            if isinstance(event_point_indices, list):
                for idx in event_point_indices:
                    if isinstance(idx, (int, np.integer)):
                        clicked_points.append(int(idx))

        # Fallback: derive selected points from lasso polygon if payload has
        # lasso geometry but no explicit selected points/indices.
        if not clicked_points and lasso_x and lasso_y:
            poly_selected = _points_in_polygon(coords_2d, lasso_x, lasso_y)
            clicked_points.extend(poly_selected)

        if not clicked_points and box_bounds is not None:
            x0, x1, y0, y1 = box_bounds
            box_mask = (
                (coords_2d[:, 0] >= x0) & (coords_2d[:, 0] <= x1) &
                (coords_2d[:, 1] >= y0) & (coords_2d[:, 1] <= y1)
            )
            box_selected = np.where(box_mask)[0].tolist()
            clicked_points.extend(box_selected)
    except TypeError:
        pass

    return {"clicked_points": sorted(set(clicked_points))}


def render_lda_weights_chart(
    weights_df,
    palette_name: str = "Tab10 (matplotlib)",
    title: str = "LDA Definition Weights (LD1)",
    corpus_labels: tuple[str, str] = ("corpus_1", "corpus_2"),
):
    """Render a horizontal bar chart of LDA weights per definition."""
    palette = get_palette(palette_name)
    colour_pos = palette[1] if len(palette) > 1 else "#56B4E9"
    colour_neg = palette[0] if len(palette) > 0 else "#E69F00"

    df = weights_df.sort_values("weight", ascending=True)
    colours = [colour_pos if w > 0 else colour_neg for w in df["weight"]]
    display_labels = [d[:60] + "…" if len(d) > 60 else d for d in df["definition"]]

    fig = go.Figure(go.Bar(
        x=df["weight"], y=display_labels, orientation="h",
        marker_color=colours, hovertext=df["definition"], hoverinfo="text+x",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="LD1 Weight", yaxis_title="",
        height=max(300, 30 * len(df)),
        margin=dict(l=250, r=30, t=50, b=50),
        annotations=[
            dict(x=0.01, y=1.05, xref="paper", yref="paper",
                 text=f"← {corpus_labels[0]}", showarrow=False, font=dict(color=colour_neg, size=11)),
            dict(x=0.99, y=1.05, xref="paper", yref="paper",
                 text=f"{corpus_labels[1]} →", showarrow=False, font=dict(color=colour_pos, size=11)),
        ],
    )

    st.plotly_chart(fig, width='stretch')
