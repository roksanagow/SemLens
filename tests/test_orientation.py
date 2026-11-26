"""The start page, the status line, autosave and continuing, warnings before work is replaced, and the packaged example."""
import json
import os

import numpy as np
import torch
from streamlit.testing.v1 import AppTest

import app.autosave as autosave
import app.preferences as preferences
from semlens.data_loading import load_from_sentences
from semlens.embeddings import embedded_usages_from_precomputed
from semlens.spaces import project_to_definition_space


def widget(at, kind, label):
    return next(w for w in getattr(at, kind) if w.label == label)


def labels(at, kind='button'):
    return [w.label for w in getattr(at, kind)]


def seeded(annotated=True, definitions=2):
    """An app with a small analysis open: embedded usages of 'bank', definitions and annotations."""
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    data = load_from_sentences([f'The bank had example {i}.' for i in range(8)], 'bank', [f'corpus_{i//4}' for i in range(8)])
    torch.manual_seed(3)
    vectors = torch.randn(len(data), 8)
    defs = torch.randn(definitions, 8)
    values = {'target_word_data': data, 'target_word': 'bank', 'embedded_usages': embedded_usages_from_precomputed(
                  data, vectors, metadata={'model_id': 'test/model', 'layer': -1, 'representation': 'word'}),
              'corpus_order': data.corpus_labels, 'corpus_labels': data.corpus_labels,
              'definitions_raw': [f'sense {i}' for i in range(definitions)],
              'definitions_formatted': [f'bank: sense {i}' for i in range(definitions)],
              'definition_embeddings': defs, 'def_space_all': project_to_definition_space(vectors, defs), 'data_revision': 1}
    for key, value in values.items():
        at.session_state[key] = value
    if annotated:
        at.session_state['_annotation_store'] = {'annotations': {0: {'money': None}, 5: {'river': None}}, 'sense_classes': ['money', 'river'],
                                                 'active_sense_class': 'money', 'history': []}
    at.run()
    return at, data


def slots():
    return [p for p in autosave.AUTOSAVE_ROOT.iterdir() if p.is_dir()]


# --- Start page ------------------------------------------------------------------------------------------

def test_the_start_page_opens_on_the_first_visit_only():
    preferences.PREFERENCES_FILE.unlink()
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert not at.exception
    assert any(t.value == 'Welcome to SemLens' for t in at.title)
    assert {'Try the example', 'Use my own texts', 'Open saved work', 'Start with my texts', 'Open a saved discovery'} <= set(labels(at))
    assert json.loads(preferences.PREFERENCES_FILE.read_text())['start_seen'] is True
    later = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert not any(t.value == 'Welcome to SemLens' for t in later.title)
    # It comes back from the sidebar, and choosing a section closes it.
    next(b for b in later.sidebar.button if b.label == 'Start page').click().run()
    assert any(t.value == 'Welcome to SemLens' for t in later.title)
    later.sidebar.radio(key='section').set_value('Discover related words').run()
    assert not any(t.value == 'Welcome to SemLens' for t in later.title) and not later.exception


def test_the_example_opens_at_once_in_explore_without_a_model():
    preferences.PREFERENCES_FILE.unlink()
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    widget(at, 'button', 'Try the example').click().run()
    assert not at.exception
    assert at.session_state['workspace'] == 'Explore' and at.session_state['section'] == 'Analyse a word'
    data, embedded = at.session_state['target_word_data'], at.session_state['embedded_usages']
    assert (data.word, len(data), len(data.corpus_labels)) == ('bubble', 350, 7)
    assert tuple(embedded.embeddings.shape) == (350, 1024) and embedded.metadata['model_id'] == 'pierluigic/xl-lexeme'
    assert embedded.metadata['target_markers'] is True and embedded.metadata['stored_precision'] == 'float16'
    assert len(at.session_state['definitions_formatted']) == 4 and tuple(at.session_state['def_space_all'].shape) == (350, 4)
    assert '_loaded_model' not in at.session_state
    assert at.get('plotly_chart')


def test_the_start_page_choices_lead_to_the_right_places():
    for label, check in [('Use my own texts', lambda s: s['workspace'] == 'Data & Model' and s['_data_source'] == 'Upload raw corpora (.txt)'),
                         ('Open saved work', lambda s: s['workspace'] == 'Data & Model' and s['_data_source'] == 'Saved analysis'),
                         ('Start with my texts', lambda s: s['section'] == 'Discover related words' and s['dv_step'] == 'Corpora'),
                         ('Open a saved discovery', lambda s: s['dv_source_choice'] == 'A saved discovery')]:
        at = AppTest.from_file('app/app.py', default_timeout=60).run()
        next(b for b in at.sidebar.button if b.label == 'Start page').click().run()
        widget(at, 'button', label).click().run()
        assert not at.exception and check(at.session_state), label
        assert not any(t.value == 'Welcome to SemLens' for t in at.title)


def test_the_packaged_example_embeddings_match_its_sentences():
    from app.ui_data import EXAMPLE_PATH, EXAMPLE_VECTORS, _example_embeddings
    from semlens.data_loading import load_from_csv
    data = load_from_csv(EXAMPLE_PATH, 'bubble', forms=['bubbles'], time_col='date')
    with np.load(EXAMPLE_VECTORS, allow_pickle=False) as saved:
        assert saved['vectors'].dtype == np.float16
    found = _example_embeddings(data)
    assert found is not None
    vectors, definitions, manifest = found
    assert vectors.shape == (len(data), 1024) and definitions.shape == (4, 1024)
    assert manifest['definitions_formatted'][0].startswith('bubble: ')
    assert EXAMPLE_VECTORS.stat().st_size < 1_000_000
    # Other sentences do not get these vectors.
    assert _example_embeddings(data.subset(list(range(10)))) is None


# --- Status line ----------------------------------------------------------------------------------------

def status_text(at):
    return ' '.join(c.value for c in at.caption)


def test_the_status_line_names_the_next_step():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert 'no usages loaded' in status_text(at) and 'choose a data source and load the usages' in status_text(at)
    at.button_group(key='workspace').set_value('Explore').run()
    widget(at, 'button', 'Load usages').click().run()
    assert at.session_state['workspace'] == 'Data & Model'
    at, data = seeded()
    text = status_text(at)
    assert '8 usages of “bank”' in text and 'embedded (model)' in text and '2 definitions' in text and '2 annotated' in text
    widget(at, 'button', 'Explore the usages').click().run()
    assert at.session_state['workspace'] == 'Explore' and 'Annotate the usages' in labels(at)
    assert 'kept as you work' in status_text(at)


def test_discover_has_a_status_line_too():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    assert 'corpora not read' in status_text(at) and 'no seeds' in status_text(at)
    at.button_group(key='dv_step').set_value('Seeds').run()
    widget(at, 'button', 'Read the corpora').click().run()
    assert at.session_state['dv_step'] == 'Corpora' and not at.exception


# --- Autosave and continuing ------------------------------------------------------------------------------

def test_work_is_kept_as_it_changes_and_the_embeddings_are_written_once():
    at, data = seeded()
    [folder] = slots()
    state = json.loads((folder / 'state.json').read_text())
    files = state['files']
    vectors = np.load(folder / files['vectors'])
    assert vectors.dtype == np.float16 and vectors.shape == (8, 8)
    assert 'annotations' not in state
    log = folder / files['annotations']
    assert autosave._read_annotations(log) == {data.usages[0].usage_id: {'money': None}, data.usages[5].usage_id: {'river': None}}
    assert state['summary'] == {'word': 'bank', 'usages': 8, 'corpora': 2, 'model': 'test/model', 'definitions': 2, 'annotated': 2, 'senses': 2}
    written = (folder / files['vectors']).stat().st_mtime_ns
    # An annotation appends one line to the annotations' file; the embeddings are not written again.
    store = at.session_state['_annotation_store']
    store['annotations'][3] = {'river': None}
    at.run()
    again = json.loads((folder / 'state.json').read_text())
    assert again['files'] == files and again['summary']['annotated'] == 3
    assert len(log.read_text().splitlines()) == 3
    assert (folder / files['vectors']).stat().st_mtime_ns == written
    # Clearing one appends a line too; reading the file gives the annotations as they are now.
    del store['annotations'][0]
    at.run()
    assert len(log.read_text().splitlines()) == 4
    assert autosave._read_annotations(log) == {data.usages[5].usage_id: {'river': None}, data.usages[3].usage_id: {'river': None}}
    assert sorted(p.name for p in folder.iterdir()) == sorted(['state.json', *[f for f in files.values() if f]])


def test_the_annotations_file_is_written_whole_again_once_it_has_grown():
    at, data = seeded()
    [folder] = slots()
    first = json.loads((folder / 'state.json').read_text())['files']['annotations']
    # As if the same usages had been changed some two thousand times: the next write replaces the file by one line per usage.
    written = at.session_state['_autosave_written']
    written['annotations'] = (written['annotations'][0], first, 2000)
    at.session_state['_annotation_store']['annotations'][1] = {'money': None}
    at.run()
    files = json.loads((folder / 'state.json').read_text())['files']
    assert files['annotations'] != first and not (folder / first).exists()
    assert len((folder / files['annotations']).read_text().splitlines()) == 3


def test_an_unfinished_last_line_is_skipped(tmp_path):
    log = tmp_path / 'annotations.jsonl'
    log.write_text('{"id": "a", "label": {"x": null}}\n{"id": "b", "label": {"y": 2}}\n{"id": "a", "lab')
    assert autosave._read_annotations(log) == {'a': {'x': None}, 'b': {'y': 2}}


def test_an_earlier_session_is_offered_and_continues_without_a_model():
    at, data = seeded()
    at.button_group(key='workspace').set_value('Definitions').run()
    original = at.session_state['embedded_usages'].embeddings
    later = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert any(m.value == '**Continue where you left off?**' for m in later.markdown)
    assert any('“bank” · 8 usages · embedded with model · 2 definitions · 2 annotated in 2 senses' in m.value for m in later.markdown)
    widget(later, 'button', 'Continue').click().run()
    assert not later.exception
    restored = later.session_state['target_word_data']
    assert [u.usage_id for u in restored.usages] == [u.usage_id for u in data.usages]
    assert torch.allclose(later.session_state['embedded_usages'].embeddings, original, atol=1e-2)
    assert later.session_state['_annotation_store']['annotations'] == {0: {'money': None}, 5: {'river': None}}
    assert later.session_state['definitions_formatted'] == ['bank: sense 0', 'bank: sense 1']
    assert later.session_state['def_space_all'] is not None and later.session_state['workspace'] == 'Definitions'
    assert '_loaded_model' not in later.session_state
    # The continued session keeps writing into the same folder, without writing the embeddings again.
    assert len(slots()) == 1
    later.session_state['_annotation_store']['annotations'][2] = {'money': None}
    later.run()
    assert json.loads((slots()[0] / 'state.json').read_text())['summary']['annotated'] == 3
    assert len(list(slots()[0].glob('vectors-*.npy'))) == 1


def test_starting_afresh_and_only_two_sessions_are_kept():
    seeded()
    seeded()
    assert len(slots()) == 2
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert sum(label == 'Continue' for label in labels(at)) == 2
    widget(at, 'button', 'No, start afresh').click().run()
    assert 'Continue' not in labels(at)
    seeded()
    assert len(slots()) == 2


def test_without_disk_space_the_embeddings_are_skipped_but_annotations_kept(monkeypatch):
    monkeypatch.setattr(autosave, '_room_for', lambda folder, size: False)
    at, data = seeded()
    [folder] = slots()
    state = json.loads((folder / 'state.json').read_text())
    assert state['files']['vectors'] is None and state['skipped'] == ['vectors']
    at.run()
    assert 'without the embeddings' in status_text(at)
    later = AppTest.from_file('app/app.py', default_timeout=60).run()
    widget(later, 'button', 'Continue').click().run()
    assert not later.exception
    assert later.session_state['embedded_usages'] is None and later.session_state['workspace'] == 'Data & Model'
    assert later.session_state['_annotation_store']['annotations'] == {0: {'money': None}, 5: {'river': None}}
    # The definitions stay as text, ready to create the space again.
    assert later.session_state['definitions_raw'] == ['sense 0', 'sense 1'] and later.session_state['def_space_all'] is None
    assert any('embed the usages again' in w.value for w in later.warning)


def test_a_folder_that_cannot_be_written_turns_autosave_off_quietly(monkeypatch, tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('')
    monkeypatch.setattr(autosave, 'AUTOSAVE_ROOT', blocked / 'autosave')
    at, _ = seeded()
    at.run()
    assert not at.exception and 'not kept automatically' in status_text(at)


def test_discover_offers_the_most_recently_changed_discovery(tmp_path):
    import app.discover as discover
    from semlens.discover.project import Discovery
    for name in ('older', 'newer'):
        folder = discover.DISCOVER_ROOT / name
        folder.mkdir(parents=True)
        (folder / 'vectors.json').write_text(json.dumps({'model_id': 'm', 'occurrences': 12}))
        (folder / 'sample.json').write_text(json.dumps({'corpora': ['a', 'b']}))
        Discovery(folder, name, seed_words=['bubble'], marks={1: True}).save()
        if name == 'older':
            os.utime(folder / 'discovery.json', (1e9, 1e9))
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    assert any('“newer” · 12 usages in 2 corpora · seeds: bubble · 1 usage marked' in m.value for m in at.markdown)
    widget(at, 'button', 'No, start afresh').click().run()
    assert not any('Continue where you left off' in m.value for m in at.markdown) and not at.exception


# --- Warnings before replacing work ---------------------------------------------------------------------------

def load_other_word(at, word='river'):
    widget(at, 'selectbox', 'Data source').set_value('Paste sentences').run()
    widget(at, 'text_input', 'Target word').set_value(word)
    at.text_area[0].set_value(f'The {word} is long.\nA {word} again.')
    at.text_area[1].set_value(f'One more {word}.')
    widget(at, 'button', 'Load usages').click().run()


def test_loading_other_usages_asks_first_when_work_is_unsaved():
    at, data = seeded()
    load_other_word(at)
    assert not at.exception and at.session_state['target_word_data'] is data
    assert any('Not saved yet: 2 definitions and the annotations of 2 usages (2 senses).' in w.value for w in at.warning)
    widget(at, 'button', 'Cancel').click().run()
    assert not any('Not saved yet' in w.value for w in at.warning)
    widget(at, 'button', 'Load usages').click().run()
    widget(at, 'button', 'Continue without saving').click().run()
    assert at.session_state['target_word_data'].word == 'river' and not at.exception


def test_saving_first_keeps_the_earlier_analysis():
    import app.ui_data as ui
    at, data = seeded()
    load_other_word(at)
    widget(at, 'button', 'Save it, then continue').click().run()
    assert not at.exception and at.session_state['target_word_data'].word == 'river'
    saved = ui.STORAGE / 'projects' / 'bank.semlens'
    assert saved.is_file() and any('Saved the earlier analysis as “bank”' in s.value for s in at.success)
    from semlens.projects import load_project
    assert load_project(saved)['annotations'] == {0: {'money': None}, 5: {'river': None}}


def test_no_question_when_the_same_word_is_loaded_again_without_definitions():
    at, data = seeded(definitions=0)
    # The same word again keeps the annotations of the usages loaded again, so there is nothing to lose.
    load_other_word(at, 'bank')
    assert not any('Not saved yet' in w.value for w in at.warning) and at.session_state['target_word_data'] is not data
    # With definitions, the question names only them.
    at, data = seeded()
    load_other_word(at, 'bank')
    assert any('Not saved yet: 2 definitions. The annotations of the usages of “bank” that are loaded again are kept' in w.value for w in at.warning)


def test_saving_marks_the_work_as_saved():
    import app.ui_data as ui
    at, data = seeded()
    widget(at, 'button', 'Prepare analysis file').click().run()
    widget(at, 'button', 'Save analysis locally').click().run()
    assert (ui.STORAGE / 'projects' / 'bank.semlens').is_file()
    load_other_word(at)
    assert not any('Not saved yet' in w.value for w in at.warning) and at.session_state['target_word_data'].word == 'river'


# --- Advanced options ---------------------------------------------------------------------------------------

def without_advanced_options():
    preferences.PREFERENCES_FILE.write_text(json.dumps({'start_seen': True}))


def test_without_advanced_options_the_model_and_score_settings_are_hidden():
    without_advanced_options()
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert not at.exception and at.session_state['advanced_options'] is False
    assert 'Language model' not in labels(at, 'selectbox') and 'Layer (-1 = final output)' not in labels(at, 'number_input')
    assert any('Model: XL-LEXEME, final layer' in c.value for c in at.caption)
    sources = widget(at, 'selectbox', 'Data source').options
    assert 'Pre-computed embeddings' not in sources and 'Saved word embeddings' not in sources and 'Saved analysis' in sources
    assert 'Vector distance' not in labels(at, 'selectbox') and 'Corpus order' not in labels(at, 'multiselect')
    # Display settings stay in the sidebar for everyone.
    assert {'Point size', 'Point opacity'} <= set(labels(at.sidebar, 'slider'))
    at.sidebar.toggle(key='advanced_options').set_value(True).run()
    assert 'Language model' in labels(at, 'selectbox')
    assert json.loads(preferences.PREFERENCES_FILE.read_text())['advanced_options'] is True


def test_score_settings_sit_next_to_the_scores_and_are_shared():
    at, data = seeded()
    assert 'Vector distance' not in labels(at.sidebar, 'selectbox')
    at.session_state['explore_view'] = 'Scores'
    at.button_group(key='workspace').set_value('Explore').run()
    distance = next(s for s in at.selectbox if s.label == 'Vector distance')
    distance.set_value('euclidean').run()
    assert at.session_state['distance_metric'] == 'euclidean'
    assert any('Euclidean distance' in c.value for c in at.caption)
    # The comparison tab has the same settings, showing the same values.
    at.session_state['explore_view'] = 'Compare two corpora'
    at.run()
    assert next(s for s in at.selectbox if s.label == 'Vector distance').value == 'euclidean'
    assert any('Distances: euclidean' in c.value for c in at.caption)
    # Corpus order is in Data & Model.
    at.button_group(key='workspace').set_value('Data & Model').run()
    assert 'Corpus order' in labels(at, 'multiselect')


def test_the_start_page_switch_sets_advanced_options():
    preferences.PREFERENCES_FILE.unlink()
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert at.session_state['advanced_options'] is False
    at.toggle(key='_advanced_start').set_value(True).run()
    assert at.session_state['advanced_options'] is True
    widget(at, 'button', 'Use my own texts').click().run()
    assert at.sidebar.toggle(key='advanced_options').value is True and 'Language model' in labels(at, 'selectbox')


def test_discover_hides_its_model_and_scoring_settings_without_advanced_options():
    without_advanced_options()
    at = AppTest.from_file('app/app.py', default_timeout=60)
    at.session_state['source_texts'] = {'a': 'The bubble of friends. ' * 6, 'b': 'The silo of friends. ' * 6}
    at.run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    at.selectbox(key='dv_source_choice').set_value('Texts loaded in Analyse a word').run()
    widget(at, 'button', 'Read the corpora').click().run()
    at.button_group(key='dv_step').set_value('Sample & embed').run()
    assert not at.exception
    assert 'Language model' not in labels(at, 'selectbox') and 'Minimum usages in a corpus' not in labels(at, 'number_input')
    assert any('Model: XLM-RoBERTa base, layer 8 of 12, target word' in c.value for c in at.caption)


def test_loading_shows_the_counts_under_the_load_button():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    widget(at, 'selectbox', 'Data source').set_value('Paste sentences').run()
    widget(at, 'text_input', 'Target word').set_value('bank')
    at.text_area[0].set_value('The bank was shut.\nTwo banks met.')
    at.text_area[1].set_value('A river bank.')
    widget(at, 'button', 'Load usages').click().run()
    assert not at.exception
    # Without opening anything: the message and the table, in Data & Model.
    assert any('Loaded 2 usages from 2 corpora' in s.value for s in at.success)
    assert any(m.value == '**Loaded: 2 usages of “bank”**' for m in at.markdown)
    counts = at.dataframe[0].value.set_index('Corpus')
    assert list(counts['Usages']) == [1, 1] and list(counts['Found in the corpus']) == [1, 1]


def test_corpora_named_by_years_are_put_in_time_order():
    from semlens.timeline import year_order
    assert year_order(['2020–2021', '1999', '2004-2009']) == ['1999', '2004-2009', '2020–2021']
    assert year_order(['1999', 'modern']) is None and year_order(['1999']) is None
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    widget(at, 'selectbox', 'Data source').set_value('Paste sentences').run()
    widget(at, 'text_input', 'Target word').set_value('bank')
    widget(at, 'text_input', 'Corpus name').set_value('2010')
    at.text_area[0].set_value('The bank was shut.')
    next(w for w in at.text_input if w.label == 'Corpus name' and w.value == 'corpus_2').set_value('1999')
    at.text_area[1].set_value('A river bank.')
    widget(at, 'button', 'Load usages').click().run()
    assert not at.exception
    assert at.session_state['corpus_order'] == ['1999', '2010'] and at.session_state['chronological'] is True


def test_discover_reads_corpora_named_by_years_in_time_order(tmp_path):
    at = AppTest.from_file('app/app.py', default_timeout=60)
    at.session_state['source_texts'] = {'2020': 'The bubble of friends. ' * 6, '1999': 'The silo of friends. ' * 6}
    at.run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    at.selectbox(key='dv_source_choice').set_value('Texts loaded in Analyse a word').run()
    widget(at, 'button', 'Read the corpora').click().run()
    assert not at.exception and at.session_state['dv_corpora'] == ['1999', '2020']


def test_the_annotation_guide_ticks_off_steps_and_can_be_turned_off():
    at, data = seeded(annotated=False)
    at.button_group(key='workspace').set_value('Annotation').run()
    assert any(m.value == '**How to annotate, step by step**' for m in at.markdown)
    assert any(m.value.startswith('**1.** **Name a sense**') for m in at.markdown)
    widget(at, 'text_input', 'New sense').set_value('money')
    widget(at, 'button', 'Add sense').click().run()
    assert any(m.value.startswith(':green[:material/check_circle:] **Name a sense**') for m in at.markdown)
    # Hidden for now: back in a later session; turned off: remembered.
    widget(at, 'button', 'Hide for now').click().run()
    assert not any(m.value == '**How to annotate, step by step**' for m in at.markdown)
    later = AppTest.from_file('app/app.py', default_timeout=60).run()
    later.session_state['workspace'] = 'Annotation'
    widget(later, 'button', 'Continue').click().run()
    assert any(m.value == '**How to annotate, step by step**' for m in later.markdown)
    widget(later, 'button', 'Don’t show this again').click().run()
    assert json.loads(preferences.PREFERENCES_FILE.read_text())['annotation_guide_off'] is True
    assert not any(m.value == '**How to annotate, step by step**' for m in later.markdown)
