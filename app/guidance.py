"""Orientation: the start page, the status line under the tabs, and the offer to continue earlier work."""
from __future__ import annotations
import streamlit as st
from app.annotation_store import get_annotations
from app.common import ADVANCED_HELP
from app.autosave import autosave_status, recoverable, restore, when

PAGES = ['Data & Model', 'Explore', 'Definitions', 'Annotation']
ABOUT = ('SemLens shows how a word is used in different collections of texts (corpora), such as different periods or genres, '
         'and helps you judge whether its meaning differs between them. A language model places every usage of the word so that '
         'similar usages sit close together, and every point and score leads back to sentences you can read.')


# --- Start page ----------------------------------------------------------------------------------------

def open_start_page_once():
    """On the very first visit, open the start page; afterwards it opens only from the sidebar."""
    if st.session_state.get('_start_checked'):
        return
    st.session_state['_start_checked'] = True
    if not st.session_state.get('start_seen'):
        st.session_state['_start_open'] = True
        st.session_state['start_seen'] = True


def open_start_page():
    st.session_state['_start_open'] = True


def close_start_page():
    st.session_state['_start_open'] = False


def start_page_open():
    return bool(st.session_state.get('_start_open'))


def _go_analyse(workspace, source=None):
    close_start_page()
    st.session_state.section = 'Analyse a word'
    st.session_state.workspace = workspace
    if source:
        st.session_state['_data_source'] = source


def _go_discover(source=None):
    close_start_page()
    st.session_state.section = 'Discover related words'
    st.session_state.dv_step = 'Corpora'
    if source:
        st.session_state.dv_source_choice = source


def _try_example():
    from app.ui_data import EXAMPLE_SOURCE, load_example, unsaved_work
    if unsaved_work('bubble'):
        # Data & Model asks first what to do with the open work.
        _go_analyse('Data & Model', EXAMPLE_SOURCE)
        return
    try:
        embedded = load_example()
    except Exception as exc:
        st.session_state['_start_error'] = str(exc)
        return
    # The status line says what was loaded; the Data & Model notice would come too late.
    st.session_state.pop('_load_notices', None)
    _go_analyse('Explore' if embedded else 'Data & Model', EXAMPLE_SOURCE)


def render_start_page():
    st.title('Welcome to SemLens')
    st.write(ABOUT)
    error = st.session_state.pop('_start_error', None)
    if error:
        st.error(error)
    if st.session_state.get('target_word_data') is None:
        render_recovery_offer('start')
    st.subheader('Where would you like to start?')
    left, right = st.columns(2, gap='medium')
    with left.container(border=True, key='start_analyse'):
        st.markdown('#### Analyse a word')
        st.caption('How one word, or a few, are used in each corpus: maps of the usages, scores, definitions and your own annotation.')
        st.button('Try the example', type='primary', icon=':material/play_arrow:', on_click=_try_example, key='start_example',
                  help='350 sentences with *bubble* from the Guardian, 1999–2026, already embedded: it opens at once in Explore.')
        st.button('Use my own texts', icon=':material/upload_file:', on_click=_go_analyse, args=('Data & Model', 'Upload raw corpora (.txt)'),
                  key='start_own', help='One plain-text file per corpus, pasted sentences, or a table of sentences.')
        st.button('Open saved work', icon=':material/folder_open:', on_click=_go_analyse, args=('Data & Model', 'Saved analysis'),
                  key='start_saved', help='An analysis saved earlier on this computer, or a .semlens file.')
    with right.container(border=True, key='start_discover'):
        st.markdown('#### Discover related words')
        st.caption('Find the words used like a few seed words, in each corpus, then open the ones you choose in Analyse a word.')
        st.button('Start with my texts', icon=':material/travel_explore:', on_click=_go_discover, key='start_discover_new')
        st.button('Open a saved discovery', icon=':material/folder_open:', on_click=_go_discover, args=('A saved discovery',),
                  key='start_discover_saved')
    # The sidebar's switch is not drawn while this page is open; this one sets the same setting.
    st.session_state['_advanced_start']=bool(st.session_state.get('advanced_options'))
    st.toggle('Advanced options',key='_advanced_start',help=ADVANCED_HELP,
              on_change=lambda:st.session_state.update(advanced_options=st.session_state['_advanced_start']))
    st.caption('Off, SemLens uses settings that suit most data, and shows fewer controls. Switch it in the sidebar at any time.')
    st.caption('This page opens by itself only on your first visit. Return to it from **Start page** in the sidebar. '
               'The **Glossary** in the sidebar explains the terms, and each **(?)** explains its control.')


# --- Status line ---------------------------------------------------------------------------------------

def status_line(items, following, go, key, note=None):
    """A line of steps, done or not, and the next step: (label, destination) for a button, or a sentence."""
    with st.container(horizontal=True, vertical_alignment='center', gap='small', key=f'{key}_status'):
        icons = {True: ':green[:material/check_circle:]', False: ':gray[:material/radio_button_unchecked:]'}
        st.caption('   '.join(f'{icons[bool(done)]} {text}' for done, text in items), width='content')
        if isinstance(following, tuple):
            st.button(following[0], icon=':material/arrow_forward:', type='tertiary', on_click=go, args=(following[1],), key=f'{key}_next')
        elif following:
            st.caption(f'**Next:** {following}', width='content')
        if note:
            st.caption(note[0], help=note[1], width='content')


def _model_name(model_id):
    from app.common import model_name
    return model_name(model_id)


def _plural(count, word):
    return f'{count:,} {word}{"" if count == 1 else "s"}'


def render_analysis_status(page, go):
    state = st.session_state
    data, embedded = state.target_word_data, state.embedded_usages
    if data is None:
        items = [(False, 'no usages loaded')]
    else:
        definitions = len(state.definitions_formatted or []) if state.def_space_all is not None else 0
        annotated = len(get_annotations())
        items = [(True, f'{_plural(len(data), "usage")} of “{data.word}”'),
                 (embedded is not None, f'embedded ({_model_name(embedded.metadata.get("model_id"))})' if embedded is not None else 'not embedded'),
                 (definitions > 0, _plural(definitions, 'definition') if definitions else 'no definitions'),
                 (annotated > 0, f'{annotated:,} annotated' if annotated else 'no annotations')]
    if data is None:
        following = 'choose a data source and load the usages, below.' if page == 'Data & Model' else ('Load usages', 'Data & Model')
    elif embedded is None:
        following = 'embed the usages, below.' if page == 'Data & Model' else ('Embed the usages', 'Data & Model')
    elif page == 'Data & Model':
        following = ('Explore the usages', 'Explore')
    elif page == 'Explore':
        following = ('Write definitions', 'Definitions') if state.def_space_all is None else ('Annotate the usages', 'Annotation')
    elif page == 'Definitions':
        following = ('Annotate the usages', 'Annotation')
    elif not get_annotations():
        following = 'name a sense, then click or lasso usages on the plot.'
    else:
        following = 'keep a named copy with **Save or export this analysis**, below.'
    status = autosave_status()
    note = None
    if status == 'on':
        note = (':material/save: kept as you work', 'SemLens keeps the open analysis on this computer as you work and offers to continue it '
                'next time. Only the latest two sessions are kept: to keep this one for good, use **Save or export this analysis** at the bottom of the page.')
    elif status == 'partial':
        note = (':material/save: kept as you work, without the embeddings', 'The disk is nearly full, so the embeddings are not kept for next time: '
                'continuing later would need them made again. Annotations, senses and definitions are kept as you work.')
    elif status == 'off':
        note = (':material/warning: not kept automatically', 'The open analysis cannot be written to SemLens’s folder: '
                f'{st.session_state.get("_autosave_off")}. Save it with **Save or export this analysis** to keep it.')
    status_line(items, following, go, 'analysis', note)


# --- Continuing earlier work -------------------------------------------------------------------------

def _offers():
    """The earlier sessions on offer, read once per session."""
    if '_recovery_offers' not in st.session_state:
        st.session_state['_recovery_offers'] = recoverable()
    return st.session_state['_recovery_offers']


def _dismiss_recovery():
    st.session_state['_recovery_dismissed'] = True


def _continue(folder):
    try:
        found = restore(folder)
    except Exception as exc:
        st.session_state['_recovery_error'] = f'The earlier work could not be opened: {exc}'
        return
    _dismiss_recovery()
    workspace = found['workspace'] if found['workspace'] in PAGES else 'Explore'
    if not found['embedded']:
        workspace = 'Data & Model'
    _go_analyse(workspace)
    if 'vectors' in found['skipped'] or not found['embedded']:
        st.session_state.setdefault('_load_notices', []).append(
            ('warning', 'The embeddings were not kept for this session, so embed the usages again below. The annotations and senses are back.'))


def _describe(summary):
    parts = [f'“{summary.get("word", "?")}”', _plural(int(summary.get('usages', 0)), 'usage')]
    if summary.get('model'):
        parts.append(f'embedded with {_model_name(summary["model"])}')
    if summary.get('definitions'):
        parts.append(_plural(int(summary['definitions']), 'definition'))
    if summary.get('annotated'):
        parts.append(f'{summary["annotated"]:,} annotated' + (f' in {_plural(int(summary["senses"]), "sense")}' if summary.get('senses') else ''))
    return ' · '.join(parts)


def render_recovery_offer(place='workspace'):
    """While nothing is loaded: offer the work of the latest earlier sessions. Never automatic."""
    if st.session_state.get('_recovery_dismissed') or st.session_state.get('target_word_data') is not None:
        return
    error = st.session_state.pop('_recovery_error', None)
    if error:
        st.error(error)
    offers = _offers()
    if not offers:
        return
    with st.container(border=True, key=f'recovery_{place}'):
        st.markdown('**Continue where you left off?**')
        for i, offer in enumerate(offers):
            with st.container(horizontal=True, vertical_alignment='center', key=f'recovery_{place}_{i}'):
                text = _describe(offer['summary'])+f' · last changed {when(offer["changed"])}'
                if 'vectors' in offer['skipped']:
                    text += ' · without its embeddings (the disk was nearly full)'
                st.write(text)
                st.button('Continue', key=f'recovery_{place}_continue_{i}', type='primary' if i == 0 else 'secondary',
                          on_click=_continue, args=(offer['folder'],))
        with st.container(horizontal=True, vertical_alignment='center', key=f'recovery_{place}_footer'):
            st.button('No, start afresh', type='tertiary', on_click=_dismiss_recovery, key=f'recovery_{place}_dismiss')
            st.caption('Continuing only reads what was kept: nothing is embedded again. '
                       'The latest two sessions are kept; starting afresh keeps them until new work replaces the older one.', width='content')
