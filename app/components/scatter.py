"""Plotly scatter plot component with hover, click-to-annotate, and corpus colouring."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from semlens.utils import (
    bold_word_in_sentence,
    format_hover_text,
    get_palette,
    split_text,
)


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
    key: str = "scatter",
) -> dict:
    """Render an interactive Plotly scatter plot in Streamlit.

    Parameters
    ----------
    enable_selection : bool
        If True, enables lasso/box selection via ``on_select="rerun"``.
        Only set this for the annotation chart — display-only charts
        should leave this False to avoid interfering with selection events.

    Returns
    -------
    dict
        ``{"clicked_points": list[int]}`` — indices of selected points
        (empty if ``enable_selection`` is False or nothing selected).
    """
    palette = get_palette(palette_name)
    unique_corpora = sorted(set(corpus_per_point))
    corpus_colour_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_corpora)}

    # Build hover text
    hover_texts = []
    for i, (sent, corp) in enumerate(zip(sentences, corpus_per_point)):
        nearest_def = definition_labels_per_point[i] if definition_labels_per_point else None
        hover_texts.append(format_hover_text(sent, target_word, corp, nearest_definition=nearest_def))

    fig = go.Figure()

    # --- One trace per corpus for clean legend ---
    for corpus_name in unique_corpora:
        mask = [i for i, c in enumerate(corpus_per_point) if c == corpus_name]
        colour = corpus_colour_map[corpus_name]

        # Check for annotations on these points
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
            customdata=mask,
            marker=dict(
                size=9,
                color=colour,
                opacity=0.8,
                line=dict(
                    width=marker_line_widths,
                    color=marker_line_colours,
                ),
            ),
        ))

    # --- Annotation overlay: show sense labels as coloured outlines ---
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
                    size=13,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color=sense_colour_map[sense_name]),
                    symbol="diamond",
                ),
            ))

    # --- Optional definition anchors ---
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
            marker=dict(
                size=12,
                color="#999999",
                symbol="star",
                line=dict(width=1, color="#333333"),
            ),
        ))

    # --- Highlight specific points (e.g. usages closest to a definition) ---
    if highlight_indices is not None and len(highlight_indices) > 0:
        fig.add_trace(go.Scatter(
            x=coords_2d[highlight_indices, 0].tolist(),
            y=coords_2d[highlight_indices, 1].tolist(),
            mode="markers",
            name="Highlighted",
            text=[hover_texts[i] for i in highlight_indices],
            hoverinfo="text",
            customdata=highlight_indices,
            marker=dict(
                size=15,
                color="rgba(0,0,0,0)",
                line=dict(width=3, color="#FF0000"),
                symbol="circle",
            ),
            showlegend=True,
        ))

    layout_kwargs = dict(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        hovermode="closest",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    if enable_selection:
        layout_kwargs["dragmode"] = "lasso"
        layout_kwargs["newselection"] = dict(mode="gradual")

    fig.update_layout(**layout_kwargs)

    # --- Render ---
    if not enable_selection:
        # Display-only: no selection events, no interference with other charts
        st.plotly_chart(fig, use_container_width=True, key=key)
        return {"clicked_points": []}

    # Selection-enabled (annotation chart only)
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",
    )

    print("DEBUG event data:", event)  # Debugging output

    # --- Parse selection events ---
    clicked_points = []
    try:
        points = event.selection.points
        for pt in points:
            cd = pt.get("customdata")
            if cd is None:
                continue
            if isinstance(cd, (list, tuple)):
                for item in cd:
                    if isinstance(item, (int, float)):
                        clicked_points.append(int(item))
            elif isinstance(cd, (int, float)):
                clicked_points.append(int(cd))
    except (AttributeError, TypeError):
        pass

    clicked_points = sorted(set(clicked_points))
    return {"clicked_points": clicked_points}


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

    display_labels = [
        d[:60] + "…" if len(d) > 60 else d
        for d in df["definition"]
    ]

    fig = go.Figure(go.Bar(
        x=df["weight"],
        y=display_labels,
        orientation="h",
        marker_color=colours,
        hovertext=df["definition"],
        hoverinfo="text+x",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="LD1 Weight",
        yaxis_title="",
        height=max(300, 30 * len(df)),
        margin=dict(l=250, r=30, t=50, b=50),
        annotations=[
            dict(
                x=0.01, y=1.05, xref="paper", yref="paper",
                text=f"← {corpus_labels[0]}",
                showarrow=False, font=dict(color=colour_neg, size=11),
            ),
            dict(
                x=0.99, y=1.05, xref="paper", yref="paper",
                text=f"{corpus_labels[1]} →",
                showarrow=False, font=dict(color=colour_pos, size=11),
            ),
        ],
    )

    st.plotly_chart(fig, use_container_width=True)
