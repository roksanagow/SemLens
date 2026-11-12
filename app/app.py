"""SemLens: explore word usages across up to ten corpora.

Run with: streamlit run app/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import streamlit as st
from semlens.utils import palette_names
from app.state import init_state
from app.common import SAMPLING_HELP,GLOSSARY,natural_sort_key
from app.ui_data import render_data,render_save_analysis
from app.ui_explore import render_full_space
from app.ui_definitions import render_definitions
from app.ui_annotations import render_annotations
from app.presentation import apply_theme
from app.components.corpus_visibility import sync_corpus_visibility

st.set_page_config(page_title='SemLens',page_icon='🔍',layout='wide')
init_state()
with st.sidebar:
    st.title('🔍 SemLens')
    st.caption('Explore meaning across corpora')
    st.radio('Theme',['Light','Dark'],horizontal=True,key='theme_mode')
    apply_theme()
    data=st.session_state.target_word_data
    if data is not None:
        st.write(f'**{data.word}** · {len(data):,} occurrences')
        for corpus in st.session_state.corpus_order:
            st.caption(f'{corpus}: {len(data.indices_for_corpus(corpus)):,}')
    with st.expander('Display & comparison settings'):
        revision=st.session_state.data_revision
        palettes=palette_names()
        st.session_state.palette_name=st.selectbox('Colour palette',palettes,index=palettes.index(st.session_state.palette_name),key=f'palette_{revision}',
            help='Colours for the corpora. The palettes marked colourblind are safe for the common forms of colour blindness. '
                 'The sequential palette runs from dark to light in corpus order, which suits corpora that follow one another in time.')
        distances=['cosine','euclidean']
        st.session_state.distance_metric=st.selectbox('Vector distance',distances,index=distances.index(st.session_state.distance_metric),key=f'distance_{revision}',
            help='Used to compare full embedding or definition-space vectors. Individual definition dimensions always use absolute distance.')
        modes={'Equal sizes':'balanced','All occurrences (unequal sizes allowed)':'unequal','Repeated equal-size samples':'repeated'}
        default=list(modes.values()).index(st.session_state.sampling_mode)
        chosen=st.selectbox('Score sampling',list(modes),index=default,help=SAMPLING_HELP,key=f'sampling_{revision}')
        st.session_state.sampling_mode=modes[chosen]
        if modes[chosen]=='repeated':
            st.session_state.sampling_repetitions=int(st.number_input('Repetitions',min_value=2,max_value=100,value=st.session_state.sampling_repetitions,key=f'reps_{revision}'))
        st.session_state.sampling_seed=int(st.number_input('Sampling seed',min_value=0,value=st.session_state.sampling_seed,key=f'seed_{revision}',
            help='Fixes which usages are drawn when samples are taken, so that scores can be reproduced. Any whole number will do.'))
        st.slider('Point size',min_value=4,max_value=18,key='point_size',help='Larger points are easier to see on a projector; smaller ones suit crowded plots.')
        st.slider('Point opacity',min_value=0.2,max_value=1.0,step=0.1,key='point_opacity',help='Lower values let overlapping points show through.')
        st.radio('Save figures as',['png','svg'],format_func=str.upper,horizontal=True,key='figure_format',
                 help='The format used by the camera button on each plot. SVG scales without blurring, for papers and posters; PNG is saved at double size, for slides.')
        if data is not None:
            with st.form(f'order_form_{revision}'):
                chronology=st.checkbox('Corpora have chronological order',value=st.session_state.chronological)
                order=st.multiselect('Corpus order',data.corpus_labels,default=st.session_state.corpus_order,
                                     help='Remove and re-add names to reorder them. Keep every corpus selected.')
                apply=st.form_submit_button('Apply corpus order')
                by_name=st.form_submit_button('Order by name',help='Sorts the corpora by name, with numbers in numeric order, so that names beginning with a year fall into time order.')
                if by_name:
                    st.session_state.corpus_order=sorted(data.corpus_labels,key=natural_sort_key)
                    st.session_state.chronological=chronology
                    st.rerun()
                if apply:
                    if len(order)==len(data.corpus_labels):
                        st.session_state.corpus_order=order
                        st.session_state.chronological=chronology
                        st.rerun()
                    else:
                        st.error('Include all corpora in the order.')
    st.caption('All corpora share the plots. Score comparisons have their own corpus selectors.')
    with st.expander('Glossary'):
        st.markdown(GLOSSARY)
    # Before any plot is drawn: apply a corpus switched on or off in a plot legend.
    sync_corpus_visibility()

# Render one workspace at a time: annotation actions do not recompute hidden plots.
pages=['Data & Model','Explore','Definitions','Annotation']
page=st.radio('Workspace',pages,horizontal=True,label_visibility='collapsed',key='workspace')
if page=='Data & Model':
    render_data()
elif page=='Explore':
    render_full_space()
elif page=='Definitions':
    render_definitions()
else:
    render_annotations()

render_save_analysis()

def go_to_page(destination):
    st.session_state.workspace=destination

destination=pages[(pages.index(page)+1)%len(pages)]
label=f'Continue to {destination}' if page!=pages[-1] else 'Return to Data & Model'
with st.container(horizontal=True,horizontal_alignment='right',key='page_continue'):
    st.button(label,icon=':material/arrow_forward:',on_click=go_to_page,args=(destination,))
