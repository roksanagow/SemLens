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
from app.presentation import apply_theme,page_tabs
from app.components.corpus_visibility import sync_corpus_visibility
from app.preferences import apply_preferences,render_preferences_note,save_preferences

SECTIONS=['Analyse a word','Discover related words']

st.set_page_config(page_title='SemLens',page_icon='🔍',layout='wide')
# Remembered settings first, so that they, not the defaults, start the session.
apply_preferences()
init_state()
st.session_state.setdefault('section',SECTIONS[0])
# The workspace is drawn only in Analyse a word: re-saved every run, so that a
# visit to Discover does not reset it (Streamlit forgets the value of a widget that is not drawn).
if 'workspace' in st.session_state:
    st.session_state.workspace=st.session_state.workspace
from app.discover import keep_settings as keep_discover_settings
keep_discover_settings()
save_preferences()
with st.sidebar:
    st.title('🔍 SemLens')
    st.caption('Explore meaning across corpora')
    section=st.radio('Section',SECTIONS,key='section',
                     help='**Analyse a word**: how one word, or a few, are used in each corpus.\n\n'
                          '**Discover related words**: find the words used like a set of seed words or seed usages, in each corpus, '
                          'then open the ones you choose in Analyse a word.')
    st.radio('Theme',['Light','Dark'],horizontal=True,key='theme_mode')
    apply_theme()

if section=='Discover related words':
    from app.discover import render_discover,render_discover_sidebar
    with st.sidebar:
        render_discover_sidebar()
        render_preferences_note()
    render_discover()
    save_preferences()
    st.stop()

with st.sidebar:
    data=st.session_state.target_word_data
    if data is not None:
        st.write(f'**{data.word}** · {len(data):,} occurrences')
        for corpus in st.session_state.corpus_order:
            st.caption(f'{corpus}: {len(data.indices_for_corpus(corpus)):,}')
    with st.expander('Display & comparison settings'):
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
    render_preferences_note()
    # Before any plot is drawn: apply a corpus switched on or off in a plot legend.
    sync_corpus_visibility()

# Render one workspace at a time: annotation actions do not recompute hidden plots.
pages=['Data & Model','Explore','Definitions','Annotation']
page=page_tabs('Workspace',pages,'workspace')
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

save_preferences()
