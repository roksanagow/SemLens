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
from app.common import GLOSSARY,ADVANCED_HELP
from app.ui_data import render_data,render_save_analysis,show_load_notices
from app.ui_explore import render_full_space
from app.ui_definitions import render_definitions
from app.ui_annotations import render_annotations
from app.presentation import apply_theme,page_tabs
from app.components.corpus_visibility import sync_corpus_visibility
from app.preferences import apply_preferences,render_preferences_note,save_preferences
from app.autosave import autosave
from app.guidance import (open_start_page_once,open_start_page,close_start_page,start_page_open,render_start_page,
                          render_analysis_status,render_recovery_offer,PAGES)

SECTIONS=['Analyse a word','Discover related words']

st.set_page_config(page_title='SemLens',page_icon='🔍',layout='wide')
# Remembered settings first, so that they, not the defaults, start the session.
apply_preferences()
init_state()
st.session_state.setdefault('section',SECTIONS[0])
# The workspace is drawn only in Analyse a word: re-saved every run, so that a
# visit to Discover does not reset it (Streamlit forgets the value of a widget that is not drawn).
# The same for the open second-level tab of Explore and of Discover's Results.
for kept in ('workspace','explore_view','dv_results_view'):
    if kept in st.session_state:
        st.session_state[kept]=st.session_state[kept]
from app.discover import keep_settings as keep_discover_settings
keep_discover_settings()
open_start_page_once()
save_preferences()


def finish_run():
    """Keep the open work and the settings. Called at the end of every run that is not interrupted."""
    autosave()
    save_preferences()


with st.sidebar:
    st.title('🔍 SemLens')
    st.caption('Explore meaning across corpora')
    st.button('Start page',icon=':material/home:',type='tertiary',on_click=open_start_page,key='open_start_page',
              help='What SemLens does, and where to start: the example, your own texts or saved work.')
    section=st.radio('Section',SECTIONS,key='section',on_change=close_start_page,
                     help='**Analyse a word**: how one word, or a few, are used in each corpus.\n\n'
                          '**Discover related words**: find the words used like a set of seed words or seed usages, in each corpus, '
                          'then open the ones you choose in Analyse a word.')
    st.radio('Theme',['Light','Dark'],horizontal=True,key='theme_mode')
    if not start_page_open():
        st.toggle('Advanced options',key='advanced_options',help=ADVANCED_HELP)
    apply_theme()

if start_page_open():
    with st.sidebar:
        render_preferences_note()
    render_start_page()
    finish_run()
    st.stop()

if section=='Discover related words':
    from app.discover import render_discover,render_discover_sidebar
    with st.sidebar:
        render_discover_sidebar()
        render_preferences_note()
    render_discover()
    finish_run()
    st.stop()

with st.sidebar:
    data=st.session_state.target_word_data
    if data is not None:
        st.write(f'**{data.word}** · {len(data):,} usages')
        for corpus in st.session_state.corpus_order:
            st.caption(f'{corpus}: {len(data.indices_for_corpus(corpus)):,}')
    with st.expander('Display settings'):
        revision=st.session_state.data_revision
        if data is not None and len(data.forms)>1:
            st.session_state['_words_mode_sidebar']=st.session_state.words_mode
            st.radio('The forms are',['forms','words'],key='_words_mode_sidebar',horizontal=True,
                     format_func={'forms':'One word','words':'Different words'}.get,
                     on_change=lambda:st.session_state.update(words_mode=st.session_state['_words_mode_sidebar']),
                     help='Different words get their own colours and comparisons; which word each form belongs to is set in Data & Model.')
        palettes=palette_names()
        st.session_state.palette_name=st.selectbox('Colour palette',palettes,index=palettes.index(st.session_state.palette_name),key=f'palette_{revision}',
            help='Colours for the corpora. The palettes marked colourblind are safe for the common forms of colour blindness. '
                 'The sequential palette runs from dark to light in corpus order, which suits corpora that follow one another in time.')
        st.slider('Point size',min_value=4,max_value=18,key='point_size',help='Larger points are easier to see on a projector; smaller ones suit crowded plots.')
        st.slider('Point opacity',min_value=0.2,max_value=1.0,step=0.1,key='point_opacity',help='Lower values let overlapping points show through.')
        st.radio('Save figures as',['png','svg'],format_func=str.upper,horizontal=True,key='figure_format',
                 help='The format used by the camera button on each plot. SVG scales without blurring, for papers and posters; PNG is saved at double size, for slides.')
    st.caption('All corpora share the plots. Score comparisons have their own corpus selectors.')
    with st.expander('Glossary'):
        st.markdown(GLOSSARY)
    render_preferences_note()
    # Before any plot is drawn: apply a corpus switched on or off in a plot legend.
    sync_corpus_visibility()

def go_to_page(destination):
    st.session_state.workspace=destination


# Render one workspace at a time: annotation actions do not recompute hidden plots.
pages=PAGES
page=page_tabs('Workspace',pages,'workspace')
render_analysis_status(page,go_to_page)
if page!='Data & Model':
    # Data & Model shows them under its load buttons, where the user just clicked.
    show_load_notices()
render_recovery_offer()
if page=='Data & Model':
    render_data()
elif page=='Explore':
    render_full_space()
elif page=='Definitions':
    render_definitions()
else:
    render_annotations()

render_save_analysis()

destination=pages[(pages.index(page)+1)%len(pages)]
label=f'Continue to {destination}' if page!=pages[-1] else 'Return to Data & Model'
with st.container(horizontal=True,horizontal_alignment='right',key='page_continue'):
    st.button(label,icon=':material/arrow_forward:',on_click=go_to_page,args=(destination,))

finish_run()
