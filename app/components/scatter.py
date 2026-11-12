"""Plotly scatter plot component with hover, click-to-annotate, and corpus colouring."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from app.presentation import plot_chart

from semlens.utils import (
    format_hover_text,
    get_palette,
    palette_for,
    wrap_hover_html,
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
    show_definition_anchors, highlight_indices, hidden=frozenset(), highlight_label="Highlighted",
):
    """Build figure with one trace per corpus (for display-only charts)."""
    fig = go.Figure()
    shown = [i for i, c in enumerate(corpus_per_point) if c not in hidden]
    if highlight_indices is not None:
        highlight_indices = [i for i in highlight_indices if corpus_per_point[i] not in hidden]

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
            meta=dict(semlens_corpus=corpus_name),
            visible="legendonly" if corpus_name in hidden else True,
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
            sense_idx = [i for i in shown if annotations.get(i) == sense_name]
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
            name="Definitions (D1, D2, …)",
            # Short labels on the plot, in the theme's text colour; the full
            # definition is in the hover and follows the definition order.
            text=[f"D{i + 1}" for i in range(len(definition_anchor_labels))],
            hovertext=[wrap_hover_html(f"Definition {i + 1}: {label}") for i, label in enumerate(definition_anchor_labels)],
            textposition="top center",
            textfont=dict(size=14),
            hoverinfo="text",
            marker=dict(size=15, color="#f59e0b", symbol="star", line=dict(width=1, color="#78350f")),
        ))

    # Highlight
    if highlight_indices is not None and len(highlight_indices) > 0:
        fig.add_trace(go.Scatter(
            x=coords_2d[highlight_indices, 0].tolist(),
            y=coords_2d[highlight_indices, 1].tolist(),
            mode="markers",
            name=highlight_label,
            text=[hover_texts[i] for i in highlight_indices],
            hoverinfo="text",
            marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(width=3, color="#FF0000"), symbol="circle"),
            showlegend=True,
        ))

    return fig


BLIND_COLOUR = "#94a3b8"


def _build_figure_annotation(
    coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
    annotations, sense_classes, interaction_mode, hidden=frozenset(),
    highlight_indices=None, highlight_label="Highlighted", highlight_colour="#ffffff", omit=frozenset(),
):
    """Build the selectable figure, with one trace per corpus.

    Every selectable point carries its position in the data as ``customdata``,
    so a selection maps back to the same usage whichever trace it came from.
    A corpus switched off in the legend keeps its legend entry; its points,
    click targets and sense markers are left out until it is switched back on.
    """
    fig = go.Figure()
    # ``omit`` holds usages left out of the picture altogether (those already
    # annotated, when the annotator asks to see only what remains).
    shown = [i for i, c in enumerate(corpus_per_point) if c not in hidden and i not in omit]

    # In click mode, add a larger transparent hit area so users can select
    # points without needing pixel-perfect clicks on tiny markers.
    if interaction_mode == "click" and shown:
        fig.add_trace(go.Scatter(
            x=coords_2d[shown, 0].tolist(),
            y=coords_2d[shown, 1].tolist(),
            mode="markers",
            name="_hit_area",
            text=[hover_texts[i] for i in shown],
            hoverinfo="text",
            customdata=shown,
            marker=dict(
                size=22,
                color="rgba(0,0,0,0.001)",
                opacity=1.0,
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            showlegend=False,
        ))

    for corpus_name in unique_corpora:
        mask = [i for i, c in enumerate(corpus_per_point) if c == corpus_name and i not in omit]
        annotated = [bool(annotations) and i in annotations for i in mask]
        fig.add_trace(go.Scatter(
            x=coords_2d[mask, 0].tolist(),
            y=coords_2d[mask, 1].tolist(),
            mode="markers",
            name=corpus_name,
            text=[hover_texts[i] for i in mask],
            hoverinfo="text",
            customdata=mask,
            meta=dict(semlens_corpus=corpus_name),
            visible="legendonly" if corpus_name in hidden else True,
            marker=dict(
                size=9,
                color=corpus_colour_map[corpus_name],
                opacity=0.8,
                line=dict(width=[2.5 if a else 0.0 for a in annotated],
                          color=["#333333" if a else "rgba(0,0,0,0)" for a in annotated]),
            ),
        ))

    # Annotation overlay — sense diamonds
    if annotations and sense_classes:
        sense_palette = get_palette("Tol Bright (colourblind)")
        sense_colour_map = {s: sense_palette[i % len(sense_palette)] for i, s in enumerate(sense_classes)}
        for sense_name in sense_classes:
            sense_idx = [i for i in shown if annotations.get(i) == sense_name]
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

    # Search matches: a ring around each matching usage that is on show. The
    # ring carries the usage's position too, so clicking it selects that usage.
    if highlight_indices:
        ringed = [i for i in highlight_indices if corpus_per_point[i] not in hidden and i not in omit]
        if ringed:
            fig.add_trace(go.Scatter(
                x=coords_2d[ringed, 0].tolist(),
                y=coords_2d[ringed, 1].tolist(),
                mode="markers",
                name=highlight_label,
                text=[hover_texts[i] for i in ringed],
                hoverinfo="text",
                customdata=ringed,
                marker=dict(size=19, color="rgba(0,0,0,0)", symbol="circle",
                            line=dict(width=2, color=highlight_colour)),
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
    highlight_label: str = "Highlighted",
    enable_selection: bool = False,
    interaction_mode: str = "lasso",
    key: str = "scatter",
    char_spans: list[tuple[int, int]] | None = None,
    corpus_order: list[str] | None = None,
    hidden_corpora: list[str] | tuple[str, ...] = (),
    blind: bool = False,
    omit_indices: list[int] | tuple[int, ...] = (),
) -> dict:
    """Render an interactive Plotly scatter plot in Streamlit.

    Parameters
    ----------
    enable_selection : bool
        If True, the figure accepts click or lasso selection. Only set for the
        annotation chart.
    hidden_corpora : list of str
        Corpora switched off in the legend. They keep their legend entry, so
        they can be switched back on, but their points are not drawn.
    blind : bool
        Draw every corpus in the same neutral colour and leave the corpus out
        of the hover text, so usages can be annotated without knowing their
        corpus. Legend entries keep their names and still switch corpora off.

    Returns
    -------
    dict
        ``{"clicked_points": list[int]}`` — selected point indices.
    """
    unique_corpora = list(dict.fromkeys(corpus_order or corpus_per_point))
    palette = palette_for(palette_name, len(unique_corpora))
    corpus_colour_map = {c: BLIND_COLOUR if blind else palette[i] for i, c in enumerate(unique_corpora)}
    hidden = frozenset(hidden_corpora) & set(unique_corpora)

    hover_texts = []
    for i, (sent, corp) in enumerate(zip(sentences, corpus_per_point)):
        nearest_def = definition_labels_per_point[i] if definition_labels_per_point else None
        hover_texts.append(format_hover_text(sent,target_word,None if blind else corp,nearest_definition=nearest_def,
                                            char_span=char_spans[i] if char_spans is not None else None))

    if enable_selection:
        fig = _build_figure_annotation(
            coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
            annotations, sense_classes, interaction_mode, hidden,
            highlight_indices, highlight_label,
            # Unlike any corpus or sense colour, and visible on either theme.
            "#ffffff" if st.session_state.get('theme_mode', 'Dark') == 'Dark' else "#111827",
            frozenset(omit_indices),
        )
    else:
        fig = _build_figure_multi_trace(
            coords_2d, corpus_per_point, hover_texts, unique_corpora, corpus_colour_map,
            annotations, sense_classes, definition_anchor_coords, definition_anchor_labels,
            show_definition_anchors, highlight_indices, hidden, highlight_label,
        )

    # The title sits at the very top and the legend below it, from the left, with
    # room for as many legend rows as there are entries, so neither covers the other.
    # Plotly gives every entry of a wrapped horizontal legend the width of the
    # longest one, so a long name means fewer entries per row.
    names = [str(trace.name or "") for trace in fig.data if trace.showlegend is not False]
    widths = [48 + 6.2 * len(name) for name in names]
    if sum(widths) <= 1100:
        legend_rows = 1
    else:
        legend_rows = -(-len(names) // max(1, int(900 // max(widths))))
    top = 52 + 24 * legend_rows
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0, xref="paper", xanchor="left",
                   y=1, yref="container", yanchor="top", pad=dict(t=12)),
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        hovermode="closest",
        hoverlabel=dict(align='left'),
        meta=dict(semlens_usage_hover=True),
        height=560 + top,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        # A vertical toolbar in the right margin: a horizontal one sits on top of
        # the legend's first row, where corpus names need to stay clickable.
        modebar=dict(orientation="v"),
        margin=dict(l=60, r=50, t=top, b=60),
    )

    # Point size and opacity come from the display settings; rings and
    # diamonds drawn around points grow with them.
    size = float(st.session_state.get('point_size', 9))
    opacity = float(st.session_state.get('point_opacity', 0.8))
    for trace in fig.data:
        if (trace.meta or {}).get('semlens_corpus') is not None:
            trace.marker.size = size
            trace.marker.opacity = opacity
        elif trace.name == '_hit_area':
            trace.marker.size = size + 13
        elif trace.marker is not None and trace.marker.size is not None and not str(trace.name).startswith('Definitions'):
            trace.marker.size = trace.marker.size + (size - 9)

    if enable_selection:
        if interaction_mode == "click":
            # Streamlit uses pan mode for point clicks. The scoped CSS below
            # replaces its four-way cursor without disabling click selection.
            fig.update_layout(dragmode="pan", clickmode="event+select")
        else:
            fig.update_layout(dragmode="lasso", clickmode="event+select")

    # --- Render ---
    if not enable_selection:
        plot_chart(fig, width='stretch', key=key)
        return {"clicked_points": []}

    event_key=f'_selection_event_{key}'
    def selection_changed():
        st.session_state[event_key]=st.session_state.get(event_key,0)+1

    # Keep click-mode cursor styling local; lasso retains its selection cursor.
    with st.container(key='annotation_click_plot' if interaction_mode=='click' else 'annotation_lasso_plot'):
        event = plot_chart(
            fig,
            width='stretch',
            key=key,
            on_select=selection_changed,
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
                pi = pt.get("point_index", pt.get("pointIndex", pt.get("pointNumber")))
                if isinstance(pi, (int, np.integer)):
                    curve = pt.get("curve_number", pt.get("curveNumber"))
                    if isinstance(curve, (int, np.integer)) and 0 <= curve < len(fig.data):
                        trace_ids = fig.data[curve].customdata
                        if trace_ids is not None and 0 <= pi < len(trace_ids):
                            clicked_points.append(int(trace_ids[pi]))
                continue
            if isinstance(cd, (list, tuple)):
                for item in cd:
                    if isinstance(item, (int, float, np.integer, np.floating)):
                        clicked_points.append(int(item))
            elif isinstance(cd, (int, float, np.integer, np.floating)):
                clicked_points.append(int(cd))

        # Alternative selection payload used by some Plotly/Streamlit versions.
        if not clicked_points and not points and isinstance(selection, dict):
            point_indices = selection.get("point_indices", [])
            if isinstance(point_indices, list):
                for idx in point_indices:
                    if isinstance(idx, (int, np.integer)):
                        clicked_points.append(int(idx))

        if not clicked_points and not points and hasattr(event, "get"):
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

    # A corpus switched off in the legend cannot be selected, including by the
    # geometric fallbacks above.
    return {"clicked_points": sorted({i for i in clicked_points
                                      if 0 <= i < len(sentences) and corpus_per_point[i] not in hidden
                                      and i not in set(omit_indices)}),
            "selection_event":(key,st.session_state.get(event_key,0))}


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
    )

    plot_chart(fig, width='stretch')
