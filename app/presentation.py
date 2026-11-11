"""Session-local display preferences and readable tables/charts."""
from __future__ import annotations
from pathlib import Path
import html
import math
import streamlit as st


def apply_theme():
    dark=st.session_state.get('theme_mode','Dark')=='Dark'
    bg,surface,text,border=('#0e1117','#1b2230','#f1f5f9','#475569') if dark else ('#ffffff','#f1f5f9','#172033','#cbd5e1')
    help_ink,help_edge,help_top,help_bottom=('#bce9d8','#49645e','#29413d','#1b2c35') if dark else ('#28644f','#b6cec2','#ffffff','#e8f2ec')
    help_hover_ink,help_hover_edge,help_hover_top,help_hover_bottom=('#edfff7','#79b79f','#355b4e','#233d40') if dark else ('#174b37','#7da78f','#f7fff9','#dceee3')
    help_popup_top,help_popup_bottom=('#25332f','#18232a') if dark else ('#ffffff','#f0f7f3')
    help_shadow='#0005' if dark else '#17203320'
    st.html(f'''<style>
    :root {{ --sl-bg:{bg}; --sl-surface:{surface}; --sl-text:{text}; --sl-border:{border}; }}
    .stApp, [data-testid="stHeader"], [data-testid="stBottom"] {{ background:{bg}; color:{text}; color-scheme:{'dark' if dark else 'light'}; }}
    [data-testid="stSidebar"] {{ background:{surface}; color:{text}; }}
    .stApp :is(h1,h2,h3,h4,h5,h6,p,label,small), .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stWidgetLabel"], .stApp summary {{ color:{text}; }}
    .stApp :is(input,textarea), .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
    .stApp [data-baseweb="select"] > div, .stApp [data-testid="stFileUploaderDropzone"],
    [data-baseweb="popover"], [role="listbox"], [role="option"] {{ background:{surface}; color:{text}; }}
    .stApp :is(input,textarea) {{ -webkit-text-fill-color:{text}; caret-color:{text}; }}
    .stApp button[kind="secondary"], .stApp button[kind="secondaryFormSubmit"] {{ background:{surface}; color:{text}; border-color:{border}; }}
    .stApp :is(details,[data-testid="stForm"]) {{ border-color:{border}; }}
    .stApp summary:hover, .stApp details[open] > summary,
    .stApp [data-testid="stNumberInput"] button,
    .stApp [data-testid="stAlert"] {{ background:{surface}; color:{text}; }}
    .stApp [data-baseweb] svg {{ color:{text}; fill:currentColor; }}
    /* Material icons and toolbar buttons otherwise retain Streamlit's own theme. */
    .stApp [data-testid="stIconMaterial"],
    .stApp button:is([kind="header"],[kind="headerNoPadding"],[kind="tertiary"],[kind="elementToolbar"]),
    .stApp [data-testid="stHeader"] svg {{ color:{text}; fill:currentColor; }}
    /* Help icons use stroked SVGs: filling them hides the question mark. */
    .stApp [data-testid="stTooltipIcon"] button {{
        display:inline-flex; align-items:center; justify-content:center;
        width:24px; height:24px; min-width:24px; padding:0;
        border:1px solid {help_edge}; border-radius:50%;
        background:linear-gradient(145deg,{help_top},{help_bottom});
        color:{help_ink}; cursor:help;
        box-shadow:inset 0 1px 0 #ffffff18,0 1px 3px {help_shadow};
        transition:color 140ms ease,border-color 140ms ease,box-shadow 140ms ease;
    }}
    .stApp [data-testid="stTooltipIcon"] svg {{
        width:18px; height:18px; color:inherit !important; fill:none !important; stroke:currentColor !important; stroke-width:2;
    }}
    .stApp [data-testid="stTooltipIcon"] svg circle {{ display:none; }}
    .stApp [data-testid="stTooltipIcon"] button:is(:hover,:focus-visible) {{
        color:{help_hover_ink}; border-color:{help_hover_edge};
        background:linear-gradient(145deg,{help_hover_top},{help_hover_bottom});
        box-shadow:inset 0 1px 0 #ffffff20,0 0 0 3px #69b58e22;
    }}
    .stApp [data-testid="stTooltipIcon"] button:focus-visible {{
        outline:2px solid {help_hover_edge}; outline-offset:3px;
    }}
    /* Streamlit mounts help popups outside .stApp, so theme them explicitly. */
    [data-baseweb="tooltip"]:has([data-testid="stTooltipContent"]) {{ background:transparent; color:{text}; }}
    [data-baseweb="tooltip"] > div:has([data-testid="stTooltipContent"]) {{
        background:linear-gradient(145deg,{help_popup_top},{help_popup_bottom});
        color:{text}; border:1px solid {help_edge}; border-radius:10px;
        box-shadow:0 8px 24px {help_shadow},inset 0 1px 0 #ffffff10;
    }}
    [data-testid="stTooltipContent"], [data-testid="stTooltipContent"] :is(p,li,strong,em) {{ color:{text}; }}
    [data-testid="stTooltipContent"] a {{ color:{help_ink}; text-decoration:underline; }}
    @media (prefers-reduced-motion:reduce) {{
        .stApp [data-testid="stTooltipIcon"] button {{ transition:none; }}
    }}
    .stApp [data-testid="stPlotlyChart"], .stApp .js-plotly-plot {{ background:{bg}; }}
    .stApp code, .stApp [data-testid="stCode"] {{ background:{surface}; color:{text}; }}
    .stApp [data-testid="stCaptionContainer"] p {{ opacity:.8; }}
    .stApp mark {{ background:#fde68a; color:#172033; }}
    .st-key-page_continue button {{ background:#15803d !important; color:white !important; border-color:#15803d !important; }}
    .st-key-page_continue button:hover {{ background:#166534 !important; }}
    .st-key-page_continue button p, .st-key-page_continue [data-testid="stIconMaterial"] {{ color:white !important; }}
    /* Click annotation uses the ordinary arrow, including over Plotly's hit layer. */
    .st-key-annotation_click_plot .js-plotly-plot .draglayer,
    .st-key-annotation_click_plot .js-plotly-plot .draglayer *,
    .st-key-annotation_click_plot .js-plotly-plot .scatterlayer .point {{ cursor:default !important; }}
    .js-plotly-plot[data-semlens-usage-hover="true"] .hoverlayer .hovertext {{ visibility:hidden; }}
    .semlens-usage-tooltip {{
        position:fixed; z-index:10000; box-sizing:border-box;
        width:max-content; max-width:min(640px,calc(100vw - 24px));
        overflow:auto; overflow-wrap:anywhere; overscroll-behavior:contain;
        padding:10px 12px; border:1px solid var(--sl-border); border-radius:6px;
        background:var(--sl-surface); color:var(--sl-text);
        font:14px/1.45 sans-serif; text-align:left;
        box-shadow:0 3px 14px #0003;
    }}
    .semlens-usage-tooltip[hidden] {{ display:none; }}
    .semlens-pair-table {{ max-height:430px; overflow-y:auto; border:1px solid var(--sl-border); border-radius:6px; }}
    .semlens-pair-table table {{ width:100%; table-layout:fixed; border-collapse:collapse; color:var(--sl-text); background:var(--sl-bg); }}
    .semlens-pair-table th {{ position:sticky; top:0; z-index:1; background:var(--sl-surface); text-align:left; font-weight:600; }}
    .semlens-pair-table th, .semlens-pair-table td {{ padding:8px; border-bottom:1px solid var(--sl-border); vertical-align:top; }}
    .semlens-pair-table td:first-child {{ font-variant-numeric:tabular-nums; }}
    .semlens-sentence-cell {{ width:100%; overflow-x:auto; white-space:pre; padding-bottom:5px; scrollbar-width:thin; scrollbar-color:var(--sl-border) var(--sl-surface); }}
    .semlens-sentence-cell:focus-visible {{ outline:2px solid var(--sl-text); outline-offset:1px; }}
    .semlens-sentence-cell::-webkit-scrollbar {{ height:8px; }}
    .semlens-sentence-cell::-webkit-scrollbar-track {{ background:var(--sl-surface); }}
    .semlens-sentence-cell::-webkit-scrollbar-thumb {{ background:var(--sl-border); border-radius:4px; }}
    </style>''')
    # Only trusted, static application code goes into this script. Corpus text
    # travels in Plotly data and is escaped/wrapped before display.
    script=(Path(__file__).parent/'components'/'usage_hover.js').read_text()
    st.html(f'<script>{script}</script>',unsafe_allow_javascript=True)


def plot_chart(fig, **kwargs):
    dark=st.session_state.get('theme_mode','Dark')=='Dark'
    text='#f1f5f9' if dark else '#172033'
    grid='#334155' if dark else '#e2e8f0'
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white',
                      font_color=text,title_font_color=text,legend_font_color=text,
                      paper_bgcolor='#0e1117' if dark else '#ffffff',
                      plot_bgcolor='#0e1117' if dark else '#ffffff')
    fig.update_xaxes(color=text,gridcolor=grid,zerolinecolor=grid)
    fig.update_yaxes(color=text,gridcolor=grid,zerolinecolor=grid)
    fig.update_traces(colorbar_tickfont_color=text,colorbar_title_font_color=text,selector=dict(type='heatmap'))
    return st.plotly_chart(fig,theme=None,**kwargs)


def readable_table(frame, *, text_columns=(), **kwargs):
    """Give complete sentences enough column width; the grid scrolls horizontally."""
    config={}
    for name in text_columns:
        if name in frame:
            # A conservative character width also accommodates non-Latin scripts.
            length=max((len(str(v).expandtabs(4)) for v in frame[name]),default=len(name))
            config[name]=st.column_config.TextColumn(name,width=max(320,16*length+40))
    dark=st.session_state.get('theme_mode','Dark')=='Dark'
    frame=frame.style.set_properties(**{'background-color':'#1b2230' if dark else '#ffffff',
                                       'color':'#f1f5f9' if dark else '#172033'})
    st.dataframe(frame,hide_index=True,width='stretch',column_config=config,**kwargs)


def sentence_pair_html(frame):
    """Fixed columns with separately scrollable cells; escape every input string."""
    rows=[]
    for row in frame.itertuples(index=False,name=None):
        distance,usage,counterpart=row
        cells=[f'<td>{float(distance):.6g}</td>']
        for label,sentence in [('Usage',usage),('Closest counterpart',counterpart)]:
            cells.append(f'<td><div class="semlens-sentence-cell" tabindex="0" role="region" aria-label="{label}: scroll horizontally">{html.escape(str(sentence))}</div></td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    return ('<div class="semlens-pair-table"><table aria-label="Ranked nearest counterparts">'
            '<colgroup><col style="width:90px"><col><col></colgroup>'
            '<thead><tr><th scope="col">Distance</th><th scope="col">Usage</th><th scope="col">Closest counterpart</th></tr></thead>'
            '<tbody>'+''.join(rows)+'</tbody></table></div>')


def sentence_pair_table(frame, *, key):
    if frame.empty:
        st.caption('No matching pairs.')
        return
    page=1
    if len(frame)>50:
        page=int(st.number_input('Table page',min_value=1,max_value=math.ceil(len(frame)/50),value=1,
                                 key=f'{key}_{len(frame)}',help='50 pairs per page. The ranking and search cover all pairs.'))
        start=(page-1)*50
        st.caption(f'Pairs {start+1:,}–{min(start+50,len(frame)):,} of {len(frame):,}')
    st.html(sentence_pair_html(frame.iloc[(page-1)*50:page*50]))
