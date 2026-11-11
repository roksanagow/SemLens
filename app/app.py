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
from app.common import SAMPLING_HELP
from app.ui_data import render_data,render_save_analysis
from app.ui_explore import render_full_space
from app.ui_definitions import render_definitions
from app.ui_annotations import render_annotations
from app.presentation import apply_theme

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
        st.session_state.palette_name=st.selectbox('Colour palette',palettes,index=palettes.index(st.session_state.palette_name),key=f'palette_{revision}')
        distances=['cosine','euclidean']
        st.session_state.distance_metric=st.selectbox('Vector distance',distances,index=distances.index(st.session_state.distance_metric),key=f'distance_{revision}',
            help='Used to compare full embedding or definition-space vectors. Individual definition dimensions always use absolute distance.')
        modes={'Equal sizes':'balanced','All occurrences (unequal sizes allowed)':'unequal','Repeated equal-size samples':'repeated'}
        default=list(modes.values()).index(st.session_state.sampling_mode)
        chosen=st.selectbox('Score sampling',list(modes),index=default,help=SAMPLING_HELP,key=f'sampling_{revision}')
        st.session_state.sampling_mode=modes[chosen]
        if modes[chosen]=='repeated':
            st.session_state.sampling_repetitions=int(st.number_input('Repetitions',min_value=2,max_value=100,value=st.session_state.sampling_repetitions,key=f'reps_{revision}'))
        st.session_state.sampling_seed=int(st.number_input('Sampling seed',min_value=0,value=st.session_state.sampling_seed,key=f'seed_{revision}'))
        if data is not None:
            with st.form(f'order_form_{revision}'):
                chronology=st.checkbox('Corpora have chronological order',value=st.session_state.chronological)
                order=st.multiselect('Corpus order',data.corpus_labels,default=st.session_state.corpus_order,
                                     help='Remove and re-add names to reorder them. Keep every corpus selected.')
                if st.form_submit_button('Apply corpus order'):
                    if len(order)==len(data.corpus_labels):
                        st.session_state.corpus_order=order
                        st.session_state.chronological=chronology
                        st.rerun()
                    else:
                        st.error('Include all corpora in the order.')
    st.caption('All corpora share the plots. Score comparisons have their own corpus selectors.')

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
