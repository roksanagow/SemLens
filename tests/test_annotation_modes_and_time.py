"""Graded and multiple-sense annotation, and continuous time."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from streamlit.testing.v1 import AppTest

from semlens.annotations import (highest_rating, label_text, normalise_annotations, normalise_label, parse_rating,
                                 with_several_senses)
from semlens.data_loading import (TargetWordData, UsageInstance, load_from_csv, load_from_sentences,
                                  times_from_corpus_names)
from semlens.embeddings import embedded_usages_from_precomputed
from semlens.projects import load_project, project_bytes
from semlens.timeline import assign_periods, parse_time, parse_times, time_from_corpus_name


# --- Annotation format -------------------------------------------------------------------------

def test_labels_read_old_and_new_forms():
    assert normalise_label('finance') == {'finance': None}
    assert normalise_label(['finance', 'social']) == {'finance': None, 'social': None}
    assert normalise_label({'finance': '3', 'social': None}) == {'finance': 3, 'social': None}
    assert normalise_label('') == {} and normalise_label(None) == {}
    assert normalise_annotations({'2': 'finance', 3: {}, 4: {'a': 2}}) == {2: {'finance': None}, 4: {'a': 2}}
    assert parse_rating(' 4 ') == 4 and parse_rating('') is None
    for bad in ('0', '2.5', 'high'):
        with pytest.raises(ValueError):
            parse_rating(bad)
    annotations = {0: {'finance': 3, 'social': 1}, 1: {'finance': None}}
    assert with_several_senses(annotations) == 1 and highest_rating(annotations) == 3
    assert label_text(annotations[0], graded=True) == 'finance (3); social (1)'
    assert label_text(annotations[0]) == 'finance; social'


def fake_streamlit(monkeypatch, *modules, interaction='Click'):
    fake = SimpleNamespace(session_state={'annotation_interaction': interaction})
    for module in modules:
        monkeypatch.setattr(module, 'st', fake)
    return fake


def test_store_replaces_one_sense_or_adds_several_and_keeps_ratings_only_when_graded(monkeypatch):
    import app.annotation_store as store
    fake_streamlit(monkeypatch, store)
    assert store.get_options() == {'multiple': False, 'graded': False, 'scale': 4, 'rating_entry': 'ask'}
    store.set_annotation(0, 'finance', 3)
    store.set_annotation(0, 'social', 2)
    assert store.get_annotations() == {0: {'social': None}}
    store.set_option('multiple', True)
    store.set_option('graded', True)
    store.set_annotation(0, 'finance', 3)
    store.set_annotation(1, 'finance', 1)
    assert store.get_annotations() == {0: {'social': None, 'finance': 3}, 1: {'finance': 1}}
    # Renaming onto an existing class merges, keeping the stronger rating.
    store.add_sense_class('social'), store.add_sense_class('finance')
    store.rename_sense_class('social', 'finance')
    assert store.get_annotations() == {0: {'finance': 3}, 1: {'finance': 1}}
    store.set_annotation(1, 'river', 4)
    store.keep_first_senses()
    assert store.get_annotations() == {0: {'finance': 3}, 1: {'finance': 1}}
    store.undo_annotations()
    assert store.get_annotations()[1] == {'finance': 1, 'river': 4}
    store.add_sense_class('river')
    store.remove_sense_class('finance')
    assert store.get_annotations() == {1: {'river': 4}}
    # A smaller scale keeps the active rating inside it.
    store.set_active_rating(4)
    store.set_option('scale', 3)
    assert store.get_active_rating() == 3


def test_clicks_toggle_and_lasso_adds_when_several_senses_are_allowed(monkeypatch):
    import app.ui_annotations as ui
    import app.annotation_store as store
    fake = fake_streamlit(monkeypatch, ui, store)
    store.set_option('multiple', True)
    lasso = {'selection_event': ('chart', 1), 'clicked_points': [2, 4]}
    fake.session_state['annotation_interaction'] = 'Lasso'
    assert ui.assign_new_selection(lasso, 'finance')
    assert ui.assign_new_selection({'selection_event': ('chart', 2), 'clicked_points': [2, 4]}, 'social')
    assert store.get_annotations() == {2: {'finance': None, 'social': None}, 4: {'finance': None, 'social': None}}
    # A single click on a usage that already has the active sense takes it off.
    fake.session_state['annotation_interaction'] = 'Click'
    assert ui.assign_new_selection({'selection_event': ('chart', 3), 'clicked_points': [2]}, 'finance')
    assert store.get_annotations()[2] == {'social': None}
    # Graded with a default rating: a click at another rating changes the rating rather than removing the sense.
    store.set_option('graded', True)
    store.set_option('rating_entry', 'default')
    assert ui.assign_new_selection({'selection_event': ('chart', 4), 'clicked_points': [4]}, 'social', 2)
    assert store.get_annotations()[4] == {'finance': None, 'social': 2}
    assert ui.assign_new_selection({'selection_event': ('chart', 5), 'clicked_points': [4]}, 'social', 2)
    assert store.get_annotations()[4] == {'finance': None}


def small_data(corpora=('old', 'new'), per=3, times=None):
    data = load_from_sentences([f'The bank had example {i}.' for i in range(per * len(corpora))], 'bank',
                               [c for c in corpora for _ in range(per)])
    if times is not None:
        usages = [UsageInstance(u.sentence, u.word, u.char_start, u.char_end, u.corpus, u.sentence_id, dict(u.metadata), t)
                  for u, t in zip(data.usages, times)]
        data = TargetWordData(data.word, usages, data.corpus_order, data.forms, 'year')
    return data


def test_frequencies_count_each_sense_and_respect_a_rating_threshold():
    from app.ui_annotations import mean_ratings, sense_frequencies
    data = small_data()
    annotations = {0: {'finance': 4, 'social': 1}, 1: {'finance': 2}, 3: {'social': None}, 4: {'finance': 3}}
    counts = sense_frequencies(data, annotations, ['finance', 'social'], ['old', 'new'])
    assert counts.loc['finance'].tolist() == [2, 1] and counts.loc['social'].tolist() == [1, 1]
    strong = sense_frequencies(data, annotations, ['finance', 'social'], ['old', 'new'], threshold=3)
    # Below the threshold a rated sense drops out; an unrated sense always counts.
    assert strong.loc['finance'].tolist() == [1, 1] and strong.loc['social'].tolist() == [0, 1]
    means, rated = mean_ratings(data, annotations, ['finance', 'social'], ['old', 'new'])
    assert means.loc['finance', 'old'] == 3 and means.loc['finance', 'new'] == 3 and np.isnan(means.loc['social', 'new'])
    assert rated.loc['social'].tolist() == [1, 0]


@pytest.mark.parametrize('options', [{'multiple': True, 'graded': False}, {'multiple': False, 'graded': True},
                                     {'multiple': True, 'graded': True}])
def test_export_and_import_round_trip_every_annotation_mode(monkeypatch, options):
    import app.ui_annotations as ui
    import app.annotation_store as store
    fake_streamlit(monkeypatch, ui, store)
    data = small_data()
    if options['multiple'] and options['graded']:
        annotations = {0: {'finance': 4, 'social': 1}, 3: {'social': 2}}
    elif options['multiple']:
        annotations = {0: {'finance': None, 'social': None}, 3: {'social': None}}
    else:
        annotations = {0: {'finance': 4}, 3: {'social': 2}}
    table = ui.annotation_table(data, annotations, {**options, 'scale': 4}, ['finance', 'social'])
    assert table.loc[0, 'sense'] == '; '.join(annotations[0])
    assert ('rating' in table.columns) == (options['graded'] and not options['multiple'])
    assert {'sense:finance', 'sense:social'} <= set(table.columns)
    exported = pd.read_csv(io.StringIO(table.to_csv(index=False)), dtype=str, keep_default_na=False)
    matched, unmatched, added, notes = ui.import_annotations(data, exported)
    assert (matched, unmatched, sorted(added)) == (2, 0, ['finance', 'social'])
    assert store.get_annotations() == annotations
    # The file switches on what it needs.
    assert store.get_options()['multiple'] == options['multiple']
    assert store.get_options()['graded'] == options['graded']
    assert len(notes) == options['multiple'] + options['graded']


def test_semicolon_senses_and_rating_column_import_from_a_hand_made_file(monkeypatch):
    import app.ui_annotations as ui
    import app.annotation_store as store
    fake_streamlit(monkeypatch, ui, store)
    data = small_data()
    table = pd.DataFrame([{'usage_id': data.usages[1].usage_id, 'sense': 'finance; social'},
                          {'usage_id': data.usages[2].usage_id, 'sense': 'river', 'rating': '6'}]).fillna('')
    matched, _, _, notes = ui.import_annotations(data, table)
    assert matched == 2 and store.get_options() == {'multiple': True, 'graded': True, 'scale': 6, 'rating_entry': 'ask'}
    assert store.get_annotations() == {1: {'finance': None, 'social': None}, 2: {'river': 6}}
    assert len(notes) == 2


# --- Saved analyses ------------------------------------------------------------------------------

def test_analysis_files_keep_ratings_options_and_times_and_old_files_still_open():
    data = small_data(times=[1999, 2000.5, None, 2010, 2011, 2012])
    embedded = embedded_usages_from_precomputed(data, torch.randn(len(data), 4))
    options = {'multiple': True, 'graded': True, 'scale': 5}
    content = project_bytes(data, embedded, annotations={0: {'finance': 5, 'social': 2}, 3: 'river'},
                            sense_classes=['finance', 'social', 'river'], annotation_options=options)
    restored = load_project(content)
    assert restored['annotations'] == {0: {'finance': 5, 'social': 2}, 3: {'river': None}}
    assert restored['annotation_options'] == options
    assert restored['data'].times == [1999.0, 2000.5, None, 2010.0, 2011.0, 2012.0]
    assert restored['data'].time_label == 'year'
    # A file written before these options existed: one sense name per usage, no times.
    import zipfile
    source = zipfile.ZipFile(io.BytesIO(content))
    manifest = json.loads(source.read('project.json'))
    manifest['annotations'] = {data.usages[1].usage_id: 'finance'}
    for key in ('annotation_options', 'time_label'):
        manifest.pop(key)
    for usage in manifest['usages']:
        usage.pop('time')
    old = io.BytesIO()
    with zipfile.ZipFile(old, 'w') as target:
        target.writestr('project.json', json.dumps(manifest))
        target.writestr('usages.npy', source.read('usages.npy'))
    reopened = load_project(old.getvalue())
    assert reopened['annotations'] == {1: {'finance': None}} and reopened['annotation_options'] is None
    assert not reopened['data'].has_time


# --- Continuous time -----------------------------------------------------------------------------

def test_times_are_read_from_numbers_and_iso_dates():
    assert parse_time('1999') == 1999.0 and parse_time(1999.25) == 1999.25 and parse_time(' ') is None
    assert parse_time('2020-01-01') == 2020.0
    assert 2020.35 < parse_time('2020-05-10') < 2020.36
    assert parse_time('2020-05-10T02:14:27Z') == pytest.approx(parse_time('2020-05-10'), abs=1e-3)
    assert parse_time('2020-07') == pytest.approx(2020 + 182 / 366)
    for bad in ('10/05/2020', 'spring', 'nan'):
        with pytest.raises(ValueError):
            parse_time(bad)
    times, unreadable = parse_times(['1999', 'x', '', '2001-06-01'])
    assert times[0] == 1999 and times[1] is None and times[2] is None and unreadable == [1]
    assert time_from_corpus_name('1999-2003') == 2001 and time_from_corpus_name('2024–2026') == 2025
    assert time_from_corpus_name('2010') == 2010 and time_from_corpus_name('early') is None


def test_periods_of_equal_length_or_equal_size_with_undated_usages_last():
    years = list(range(1999, 2027))
    labels, order = assign_periods(years, 5, 'width')
    assert order == ['1999–2004', '2005–2010', '2011–2016', '2017–2022', '2023–2026']
    assert labels[years.index(2004)] == '1999–2004' and labels[years.index(2005)] == '2005–2010'
    labels, order = assign_periods([2000.2, 2000.9, 2003.5, None, 2008.1], 2, 'width')
    # Dates are grouped by calendar year, and usages without a time come last.
    assert order == ['2000–2004', '2005–2008', 'No time'] and labels[3] == 'No time'
    labels, order = assign_periods([0.1, 0.4, 0.6, 0.9], 2, 'width')
    assert order == ['0.1–0.5', '0.5–0.9'] and labels == ['0.1–0.5', '0.1–0.5', '0.5–0.9', '0.5–0.9']
    labels, order = assign_periods(list(range(2000, 2012)), 3, 'count')
    assert [labels.count(name) for name in order] == [4, 4, 4]
    assert assign_periods([2005, 2005], 4)[1] == ['2005']
    with pytest.raises(ValueError, match='No usage has a usable time'):
        assign_periods([None, None])


def test_csv_with_a_time_column_keeps_times_and_can_form_periods_without_a_corpus_column():
    rows = 'sentence,year,source\nA bank in 1999.,1999,x\nA bank in 2005.,2005,y\nAnother bank.,,z\nA bank in 2020.,2020,x\n'
    with pytest.raises(ValueError, match='choose a time column'):
        load_from_csv(io.StringIO(rows), 'bank')
    data = load_from_csv(io.StringIO(rows), 'bank', time_col='year', periods=2)
    assert data.times == [1999.0, 2005.0, None, 2020.0] and data.time_label == 'year'
    assert data.corpus_labels == ['1999–2009', '2010–2020', 'No time']
    assert data.usages[0].metadata == {'year': '1999', 'source': 'x'}
    with_corpus = 'sentence,corpus,date\nA bank.,early,2001-03-04\nA bank.,late,2019-11-30\n'
    data = load_from_csv(io.StringIO(with_corpus), 'bank', time_col='date')
    assert data.corpus_labels == ['early', 'late'] and 2001 < data.times[0] < 2002
    # Times are not part of a usage's identity, so annotations still match.
    plain = load_from_csv(io.StringIO(with_corpus), 'bank')
    assert [u.usage_id for u in plain.usages] == [u.usage_id for u in data.usages]


def test_corpus_names_can_stand_for_times():
    data = small_data(('1999-2003', '2010'))
    timed, unreadable = times_from_corpus_names(data)
    assert unreadable == [] and timed.times == [2001.0] * 3 + [2010.0] * 3 and timed.time_label == 'corpus'
    same, unreadable = times_from_corpus_names(small_data(('1999', 'recent')))
    assert unreadable == ['recent'] and not same.has_time


# --- The app -------------------------------------------------------------------------------------

def timed_app(blind=False):
    at = AppTest.from_file('app/app.py', default_timeout=30).run()
    data = small_data(('old', 'new'), per=6, times=[1990, 1991, 1992, 1993, 1994, 1995, 2010, 2011, 2012, 2013, 2014, None])
    torch.manual_seed(1)
    for k, v in {'target_word_data': data, 'target_word': 'bank',
                 'embedded_usages': embedded_usages_from_precomputed(data, torch.randn(len(data), 6)),
                 'corpus_order': data.corpus_labels, 'corpus_labels': data.corpus_labels, 'data_revision': 1}.items():
        at.session_state[k] = v
    return at, data


def chart(at):
    return json.loads(at.get('plotly_chart')[0].proto.spec)


def widget(at, kind, label):
    return next(w for w in getattr(at, kind) if w.label == label)


def test_over_time_view_puts_time_on_the_x_axis_with_a_moving_average():
    at, data = timed_app()
    widget(at, 'radio', 'Workspace').set_value('Explore').run()
    view = widget(at, 'selectbox', 'Plot view')
    assert 'Over time' in view.options
    view.set_value('Over time').run()
    assert not at.exception
    fig = chart(at)
    assert fig['layout']['xaxis']['title']['text'] == 'year'
    xs = {i: x for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus') for i, x in zip(t['customdata'], t['x'])}
    for i, time in enumerate(data.times):
        if time is None:
            assert xs[i] is None
        else:
            assert abs(xs[i] - time) <= 0.3 * 1 + 1e-9
    assert any(t['name'] == 'Moving average' for t in fig['data'])
    assert any('1 of 12 usages have no time' in c.value for c in at.caption)
    # Hover text shows the time.
    texts = [text for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus') for text in t['text']]
    assert any('year: 1990' in text for text in texts)


def test_points_can_be_coloured_by_time_and_time_views_are_hidden_while_blind():
    at, data = timed_app()
    widget(at, 'radio', 'Workspace').set_value('Explore').run()
    widget(at, 'radio', 'Colour points by').set_value('Time').run()
    fig = chart(at)
    assert fig['layout']['coloraxis']['colorscale'] and fig['layout']['coloraxis']['cmin'] == 1990
    corpus_traces = [t for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus') and not t['meta'].get('semlens_legend_entry')]
    assert all(t['marker']['coloraxis'] == 'coloraxis' and t['showlegend'] is False for t in corpus_traces)
    # Grey legend entries stand in for the corpora, grouped with their points.
    entries = [t for t in fig['data'] if (t.get('meta') or {}).get('semlens_legend_entry')]
    assert [t['name'] for t in entries] == data.corpus_labels
    assert all(t['legendgroup'] == t['name'] for t in entries + corpus_traces)
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    widget(at, 'toggle', 'Blind annotation').set_value(True).run()
    assert 'Over time' not in widget(at, 'selectbox', 'Plot view').options
    assert not any(r.label == 'Colour points by' for r in at.radio)
    texts = [text for t in chart(at)['data'] for text in (t.get('text') or [])]
    assert not any('year:' in text for text in texts)


def test_annotation_options_are_off_and_tucked_away_until_switched_on():
    at, data = timed_app()
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    widget(at, 'text_input', 'New sense class').set_value('finance')
    widget(at, 'button', 'Add class').click().run()
    assert any(e.label == 'Annotation options' for e in at.expander)
    assert widget(at, 'checkbox', 'Allow several senses per usage').value is False
    assert widget(at, 'checkbox', 'Graded annotation').value is False
    assert not any(n.label == 'Points on the scale' for n in at.number_input)
    widget(at, 'checkbox', 'Graded annotation').check().run()
    assert not at.exception
    assert widget(at, 'number_input', 'Points on the scale').value == 4
    assert at.session_state['_annotation_store']['options']['graded'] is True
    widget(at, 'number_input', 'Points on the scale').set_value(6).run()
    assert at.session_state['_annotation_store']['options']['scale'] == 6
    # The options survive a visit to another workspace.
    widget(at, 'radio', 'Workspace').set_value('Explore').run()
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert widget(at, 'checkbox', 'Graded annotation').value is True
    assert widget(at, 'number_input', 'Points on the scale').value == 6


def test_switching_off_several_senses_asks_before_dropping_any():
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {0: {'finance': None, 'social': None}}, 'sense_classes': ['finance', 'social'],
                                             'active_sense_class': 'finance', 'history': [],
                                             'options': {'multiple': True, 'graded': False, 'scale': 4}}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    widget(at, 'checkbox', 'Allow several senses per usage').uncheck().run()
    assert not at.exception
    assert at.session_state['_annotation_store']['options']['multiple'] is True
    assert any('have more than one sense' in w.value for w in at.warning)
    widget(at, 'button', 'Keep only the first sense of each usage').click().run()
    assert at.session_state['_annotation_store']['annotations'] == {0: {'finance': None}}
    assert at.session_state['_annotation_store']['options']['multiple'] is False
    assert widget(at, 'checkbox', 'Allow several senses per usage').value is False


def test_example_data_loads_with_its_dates_as_times():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    source = widget(at, 'selectbox', 'Data source')
    source.set_value(next(o for o in source.options if o.startswith('Example'))).run()
    widget(at, 'button', 'Load the example').click().run()
    assert not at.exception
    data = at.session_state['target_word_data']
    assert data.time_label == 'date' and all(t is not None for t in data.times) and len(data.corpus_labels) == 7


def test_list_edits_in_every_annotation_mode():
    from app.ui_annotations import NO_SENSE, UNRATED, list_edits
    text = {'Before': ['a', 'b'], 'Target': ['bank', 'bank'], 'After': ['c', 'd']}
    # One sense, ungraded.
    table = pd.DataFrame({'Sense': ['finance', NO_SENSE], **text}, index=[3, 7])
    edited = table.copy(); edited.loc[7, 'Sense'] = 'river'; edited.loc[3, 'Sense'] = NO_SENSE
    assert list_edits(table, edited, {3: {'finance': None}}, {}, False, False, 4) == {3: {}, 7: {'river': None}}
    # One sense, graded: a newly chosen sense without a rating takes the active rating.
    table = pd.DataFrame({'Sense': ['finance', NO_SENSE], 'Rating': ['2', NO_SENSE], **text}, index=[3, 7])
    edited = table.copy(); edited.loc[7, 'Sense'] = 'river'; edited.loc[3, 'Rating'] = '4'
    assert list_edits(table, edited, {3: {'finance': 2}}, {}, False, True, 3) == {3: {'finance': 4}, 7: {'river': 3}}
    # Several senses, ungraded: tick boxes, keeping the order the senses were given in.
    keys = {'sense:0': 'finance', 'sense:1': 'social'}
    table = pd.DataFrame({'sense:0': [True, False], 'sense:1': [False, False], **text}, index=[3, 7])
    edited = table.copy(); edited.loc[3, 'sense:1'] = True; edited.loc[7, 'sense:0'] = True
    assert list_edits(table, edited, {3: {'finance': None}}, keys, True, False, 4) == {3: {'finance': None, 'social': None}, 7: {'finance': None}}
    # Several senses, graded: a rating per sense, ✓ for a sense without a rating.
    table = pd.DataFrame({'sense:0': [UNRATED, NO_SENSE], 'sense:1': ['2', NO_SENSE], **text}, index=[3, 7])
    edited = table.copy(); edited.loc[3, 'sense:0'] = '3'; edited.loc[7, 'sense:1'] = UNRATED
    assert list_edits(table, edited, {3: {'social': 2, 'finance': None}}, keys, True, True, 4) == {3: {'social': 2, 'finance': 3}, 7: {'social': None}}
    assert list_edits(table, table.copy(), {}, keys, True, True, 4) == {}


def test_list_shows_a_column_per_sense_when_several_senses_are_allowed():
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {0: {'finance': 3, 'social': None}, 2: {'social': 1}},
                                             'sense_classes': ['finance', 'social'], 'active_sense_class': 'finance', 'history': [],
                                             'options': {'multiple': True, 'graded': True, 'scale': 4}}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    at.session_state['annotation_list'] = True
    at.run()
    assert not at.exception
    editor = next(frame.value for frame in at.dataframe if 'Target' in frame.value.columns)
    assert list(editor.columns[:2]) == ['sense:0', 'sense:1']
    assert list(editor['sense:0'][:3]) == ['3', '—', '—'] and list(editor['sense:1'][:3]) == ['✓', '—', '1']


def test_typed_ratings_wait_for_the_rating_box(monkeypatch):
    import app.ui_annotations as ui
    import app.annotation_store as store
    fake = fake_streamlit(monkeypatch, ui, store, interaction='Lasso')
    store.set_option('graded', True)
    assert store.get_options()['rating_entry'] == 'ask'
    assert ui.assign_new_selection({'selection_event': ('chart', 1), 'clicked_points': [1, 3]}, 'finance')
    # Nothing is assigned until a rating is typed.
    assert store.get_annotations() == {}
    assert fake.session_state['_rating_request'] == {'indices': [1, 3], 'sense': 'finance'}
    assert ui.apply_assignment([1, 3], 'finance', 2)
    assert store.get_annotations() == {1: {'finance': 2}, 3: {'finance': 2}}


def test_rating_box_saves_a_typed_rating_or_removes_the_sense():
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {2: {'finance': 1, 'social': None}}, 'sense_classes': ['finance', 'social'],
                                             'active_sense_class': 'finance', 'history': [],
                                             'options': {'multiple': True, 'graded': True, 'scale': 4, 'rating_entry': 'ask'}}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert not any(w.label == 'Default rating' for w in at.get('button_group'))
    at.session_state['_rating_request'] = {'indices': [0, 1], 'sense': 'finance'}
    at.run()
    assert not at.exception
    assert any(t.label.startswith('How clearly') and '1 = barely, 4 = clearly' in t.label for t in at.text_input)
    # Saving without a rating, or with one off the scale, asks again.
    next(b for b in at.button if b.label == 'Save rating').click().run()
    assert any('Type a rating' in e.value for e in at.error)
    next(t for t in at.text_input if t.label.startswith('How clearly')).set_value('7')
    next(b for b in at.button if b.label == 'Save rating').click().run()
    assert any('whole number from 1 to 4' in e.value for e in at.error)
    next(t for t in at.text_input if t.label.startswith('How clearly')).set_value('3')
    next(b for b in at.button if b.label == 'Save rating').click().run()
    assert not at.exception
    assert '_rating_request' not in at.session_state
    labels = at.session_state['_annotation_store']['annotations']
    assert labels[0] == {'finance': 3} and labels[1] == {'finance': 3}
    # A usage that already has the sense can lose it from the same box.
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {2: {'finance': 1, 'social': None}}, 'sense_classes': ['finance', 'social'],
                                             'active_sense_class': 'finance', 'history': [],
                                             'options': {'multiple': True, 'graded': True, 'scale': 4, 'rating_entry': 'ask'}}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    at.session_state['_rating_request'] = {'indices': [2], 'sense': 'finance'}
    at.run()
    assert any('already has “finance”, rated 1' in c.value for c in at.caption)
    next(b for b in at.button if b.label == 'Remove “finance” from this usage').click().run()
    assert at.session_state['_annotation_store']['annotations'][2] == {'social': None}


def test_default_rating_mode_shows_the_rating_buttons():
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {}, 'sense_classes': ['finance'], 'active_sense_class': 'finance', 'history': [],
                                             'options': {'multiple': False, 'graded': True, 'scale': 4, 'rating_entry': 'default'}}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert not at.exception
    assert any(w.label == 'Default rating' for w in at.get('button_group'))
    assert widget(at, 'radio', 'Giving ratings').value == 'default'


def test_frequency_chart_stacks_shares_or_totals():
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {0: {'finance': None, 'social': None}, 1: {'finance': None}, 6: {'social': None}},
                                             'sense_classes': ['finance', 'social'], 'active_sense_class': 'finance', 'history': [],
                                             'options': {'multiple': True, 'graded': False, 'scale': 4, 'rating_entry': 'ask'}}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    bars = next(json.loads(c.proto.spec) for c in at.get('plotly_chart') if json.loads(c.proto.spec)['data'][0]['type'] == 'bar')
    assert bars['layout']['barmode'] == 'stack'
    assert [t['y'] for t in bars['data']] == [[1.0, 0.0], [0.5, 1.0]]
    widget(at, 'checkbox', 'Show totals instead of shares in the chart').check().run()
    bars = next(json.loads(c.proto.spec) for c in at.get('plotly_chart') if json.loads(c.proto.spec)['data'][0]['type'] == 'bar')
    assert bars['layout']['barmode'] == 'stack' and [t['y'] for t in bars['data']] == [[2, 0], [1, 1]]
    assert bars['layout']['yaxis']['title']['text'] == 'Annotated usages'


def test_model_notes_explain_sources_and_rough_sizes(monkeypatch, tmp_path):
    import app.ui_data as ui
    assert ui.normalise_model_reference('https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main') == 'FacebookAI/xlm-roberta-base'
    assert 'Hugging Face' in ui.MODEL_HELP and 'GitHub' in ui.MODEL_HELP and 'per billion parameters' in ui.MODEL_HELP
    assert '4.5 GB of disk space' in ui._model_size_note('pierluigic/xl-lexeme')
    folder = tmp_path / 'model'; folder.mkdir(); (folder / 'model.safetensors').write_bytes(b'0' * 2_000_000)
    assert '0.0 GB' in ui._model_size_note(str(folder)) or 'GB of model weights' in ui._model_size_note(str(folder))
    monkeypatch.setattr(ui, '_hub_model_size', lambda model_id: (1_500_000_000, 0, 6_000_000_000))
    note = ui._model_size_note('someone/big-model')
    assert '1.5 billion parameters' in note and '6.0 GB download' in note and '12.0 GB of disk space' in note
    monkeypatch.setattr(ui, '_hub_model_size', lambda model_id: None)
    assert 'Could not look this model up' in ui._model_size_note('someone/offline')


# --- Loading previous annotations -----------------------------------------------------------------

def test_usages_are_found_by_id_then_position_then_sentence():
    from app.ui_annotations import usage_finder
    data = load_from_sentences(['A bank and a bank.', 'The bank.', 'Another bank here.'], 'bank', ['old', 'old', 'new'])
    find = usage_finder(data)
    first, second, third, fourth = data.usages
    assert find(usage_id=third.usage_id) == 2
    assert find(sentence=first.sentence, corpus='old', start=second.char_start, end=second.char_end) == 1
    # Another corpus name (periods formed differently): the sentence and position still identify it.
    assert find(sentence=first.sentence, corpus='1999–2004', start=str(first.char_start), end=str(first.char_end)) == 0
    # The sentence alone works only when it holds the word once.
    assert find(sentence='The bank.') == 2 and find(sentence=first.sentence) is None
    assert find(usage_id='unknown', sentence='Not loaded.') is None


def test_annotations_of_a_saved_analysis_load_onto_regrouped_usages(monkeypatch):
    import app.ui_annotations as ui
    import app.annotation_store as store
    fake_streamlit(monkeypatch, ui, store)
    saved = small_data(('1999-2003', '2004-2009'))
    embedded = embedded_usages_from_precomputed(saved, torch.randn(len(saved), 4))
    snapshot = load_project(project_bytes(saved, embedded, annotations={0: {'finance': 3}, 4: 'social'},
                                          sense_classes=['finance', 'social', 'literal'],
                                          annotation_options={'multiple': False, 'graded': True, 'scale': 5}))
    # The same sentences, now grouped into other periods.
    regrouped = small_data(('early', 'late'))
    store.set_annotation(1, 'river')
    matched, unmatched, added, notes = ui.load_analysis_annotations(regrouped, snapshot)
    assert (matched, unmatched) == (2, 0) and added == ['finance', 'social', 'literal']
    assert store.get_annotations() == {0: {'finance': 3}, 1: {'river': None}, 4: {'social': None}}
    assert store.get_options()['graded'] and store.get_options()['scale'] == 5
    store.undo_annotations()
    assert store.get_annotations() == {1: {'river': None}}
    ui.load_analysis_annotations(regrouped, snapshot, replace=True)
    assert store.get_annotations() == {0: {'finance': 3}, 4: {'social': None}}


def saved_analysis(folder, annotations):
    data = small_data(('old', 'new'), per=6, times=[1990, 1991, 1992, 1993, 1994, 1995, 2010, 2011, 2012, 2013, 2014, None])
    torch.manual_seed(1)
    embedded = embedded_usages_from_precomputed(data, torch.randn(len(data), 6))
    (folder / 'projects').mkdir(parents=True, exist_ok=True)
    path = folder / 'projects' / 'earlier work.semlens'
    path.write_bytes(project_bytes(data, embedded, annotations=annotations, sense_classes=['finance', 'social']))
    return path


def test_annotation_tab_opens_a_saved_analysis_when_nothing_is_loaded(monkeypatch, tmp_path):
    import app.ui_annotations as ui
    monkeypatch.setattr(ui, 'STORAGE', tmp_path)
    saved_analysis(tmp_path, {0: 'finance', 7: 'social'})
    at = AppTest.from_file('app/app.py', default_timeout=30).run()
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert not at.exception
    assert any('Annotation needs usages' in i.value for i in at.info)
    assert 'earlier work' in widget(at, 'selectbox', 'Saved analysis').format_func(widget(at, 'selectbox', 'Saved analysis').value)
    widget(at, 'button', 'Open analysis').click().run()
    assert not at.exception
    assert widget(at, 'radio', 'Workspace').value == 'Annotation'
    assert len(at.session_state['target_word_data']) == 12
    assert at.session_state['_annotation_store']['annotations'] == {0: {'finance': None}, 7: {'social': None}}
    assert any('Opened “earlier work”' in m.value for m in at.success)


def test_annotation_tab_loads_previous_annotations_onto_the_loaded_usages(monkeypatch, tmp_path):
    import app.ui_annotations as ui
    monkeypatch.setattr(ui, 'STORAGE', tmp_path)
    saved_analysis(tmp_path, {0: 'finance', 7: {'social': 2}})
    at, data = timed_app()
    at.session_state['_annotation_store'] = {'annotations': {3: {'finance': None}}, 'sense_classes': ['finance'],
                                             'active_sense_class': 'finance', 'history': []}
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert any(e.label == 'Load previous annotations' for e in at.expander)
    assert widget(at, 'checkbox', 'Replace the annotations made so far').value is False
    widget(at, 'button', 'Load annotations').click().run()
    assert not at.exception
    store = at.session_state['_annotation_store']
    assert store['annotations'] == {0: {'finance': None}, 3: {'finance': None}, 7: {'social': 2}}
    assert store['sense_classes'] == ['finance', 'social'] and store['options']['graded'] is True
    assert any(m.value.startswith('Loaded 2 annotations') and 'graded annotation is now on' in m.value for m in at.success)


# --- Different words -------------------------------------------------------------------------------

def word_data():
    sentences = ['A bubble here.', 'Two bubbles there.', 'A balloon went up.', 'Balloons popped.', 'Bubble tea.', 'The bubble grew.',
                 'A red balloon.', 'Bubbles everywhere.']
    return load_from_sentences(sentences, 'bubble', ['old', 'old', 'old', 'new', 'new', 'new', 'new', 'old'], forms=['bubbles', 'balloon*'])


def test_each_usage_belongs_to_the_entry_it_matched_and_entries_can_be_merged():
    from semlens.words import OTHER_FORMS, entry_counts, entry_of, regroup, usage_words, word_names
    entries = ['bubble', 'bubbles', 'balloon*', 'ball*']
    assert entry_of('bubble', entries) == 'bubble' and entry_of('Bubbles', entries) == 'bubbles'
    # The longest wildcard wins, and a form matching nothing has no entry.
    assert entry_of('Balloons', entries) == 'balloon*' and entry_of('balls', entries) == 'ball*' and entry_of('froth', entries) is None
    assert word_names(entries, {'bubbles': 'Bubble'}) == {'bubble': 'bubble', 'bubbles': 'bubble', 'balloon*': 'balloon', 'ball*': 'ball'}
    data = word_data()
    words, order = usage_words(data)
    assert order == ['bubble', 'bubbles', 'balloon'] and words[:4] == ['bubble', 'bubbles', 'balloon', 'balloon']
    words, order = usage_words(data, {'bubbles': 'bubble'})
    assert order == ['bubble', 'balloon'] and words.count('bubble') == 5
    assert entry_counts(data, data.forms) == {'bubble': 3, 'bubbles': 2, 'balloon*': 3}
    odd = load_from_sentences(['Some froth.'], 'bubble', ['old'], forms=['froth'])
    assert usage_words(odd, {})[0] == ['froth']
    assert usage_words(TargetWordData('bubble', odd.usages, forms=[]))[0] == [OTHER_FORMS]
    # Regrouping keeps each usage's identity.
    grouped = regroup(data, words, order, [0, 2, 3])
    assert grouped.corpus_labels == ['bubble', 'balloon']
    assert [u.usage_id for u in grouped.usages] == [data.usages[i].usage_id for i in (0, 2, 3)]


def word_app(group_by='corpus', words_shown=None, corpora_shown=None, annotations=None):
    at = AppTest.from_file('app/app.py', default_timeout=30).run()
    data = word_data()
    torch.manual_seed(2)
    vectors = torch.randn(len(data), 6)
    for k, v in {'target_word_data': data, 'target_word': 'bubble', 'embedded_usages': embedded_usages_from_precomputed(data, vectors),
                 'corpus_order': data.corpus_labels, 'corpus_labels': data.corpus_labels, 'data_revision': 1,
                 'words_mode': 'words', 'word_mapping': {'bubbles': 'bubble'}, 'group_by': group_by,
                 'words_shown': words_shown, 'corpora_shown': corpora_shown}.items():
        at.session_state[k] = v
    if annotations is not None:
        at.session_state['_annotation_store'] = {'annotations': annotations, 'sense_classes': ['finance', 'literal'],
                                                 'active_sense_class': 'finance', 'history': []}
    return at, data


def groups_in(fig):
    return {t['meta']['semlens_corpus']: sorted(t['customdata']) for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus')}


def test_the_switch_is_global_and_groups_every_workspace_by_word():
    at, data = word_app()
    widget(at, 'radio', 'Workspace').set_value('Explore').run()
    assert not at.exception
    # A corpus pair chosen before switching must not linger among the words.
    widget(at, 'selectbox', 'From corpus').set_value('new').run()
    assert widget(at, 'radio', 'Colour and compare by').value == 'corpus'
    assert set(groups_in(chart(at))) == {'old', 'new'}
    assert any(s.value == 'Words by corpus' for s in at.subheader)
    widget(at, 'radio', 'Colour and compare by').set_value('word').run()
    assert not at.exception
    assert at.session_state['group_by'] == 'word'
    assert set(groups_in(chart(at))) == {'bubble', 'balloon'}
    assert widget(at, 'selectbox', 'From word').options == ['bubble', 'balloon']
    assert widget(at, 'selectbox', 'From word').value == 'bubble' and widget(at, 'selectbox', 'To word').value == 'balloon'
    assert any(s.value == 'Compare words' for s in at.subheader)
    # The same choice in another workspace.
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert not at.exception
    assert widget(at, 'radio', 'Colour and compare by').value == 'word'
    assert set(groups_in(chart(at))) == {'bubble', 'balloon'}
    assert widget(at, 'multiselect', 'Corpora shown').value == ['old', 'new']


def test_showing_one_word_compares_it_across_corpora_and_hover_names_the_word():
    at, data = word_app(words_shown=['balloon'])
    widget(at, 'radio', 'Workspace').set_value('Explore').run()
    assert not at.exception
    fig = chart(at)
    # Only balloon's usages, still grouped by corpus; usage IDs in the download are the originals.
    assert sum(len(v) for v in groups_in(fig).values()) == 3
    texts = [text for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus') for text in t['text']]
    assert all('Word: balloon' in text for text in texts)
    assert widget(at, 'multiselect', 'Words shown').value == ['balloon']


def test_clicks_and_frequencies_in_a_filtered_view_use_the_loaded_usages(monkeypatch):
    import app.grouping as grouping
    at, data = word_app(group_by='word', corpora_shown=['new'], annotations={3: {'finance': None}, 0: {'literal': None}})
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    assert not at.exception
    fig = chart(at)
    # Only the new corpus is drawn, so view positions differ from the loaded ones.
    assert sum(len(v) for t, v in groups_in(fig).items()) == 4
    sense = next(t for t in fig['data'] if t['name'] == '⬡ finance')
    view_positions = sense['customdata']
    assert len(view_positions) == 1
    # Sense frequencies are by word, over the shown corpus only.
    assert any(s.value == 'Sense frequencies by word' for s in at.subheader)
    fake = SimpleNamespace(session_state={'words_mode': 'words', 'word_mapping': {'bubbles': 'bubble'}, 'group_by': 'word',
                                          'corpora_shown': ['new'], 'corpus_order': ['old', 'new']})
    monkeypatch.setattr(grouping, 'st', fake)
    import app.common as common
    monkeypatch.setattr(common, 'st', fake)
    view = grouping.current_view(data)
    assert view.indices == [3, 4, 5, 6] and view.to_original([0, 3]) == [3, 6]
    assert view.annotations({3: {'finance': None}, 0: {'literal': None}}) == {0: {'finance': None}}
    assert view.hover_notes() == ['Corpus: new'] * 4 and view.hover_notes(hide_corpus=True) is None


def test_blind_annotation_keeps_word_colours_but_hides_corpora():
    at, data = word_app(group_by='word')
    widget(at, 'radio', 'Workspace').set_value('Annotation').run()
    widget(at, 'toggle', 'Blind annotation').set_value(True).run()
    assert not at.exception
    fig = chart(at)
    colours = {t['meta']['semlens_corpus']: t['marker']['color'] for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus')}
    assert len(set(colours.values())) == 2
    texts = [text for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus') for text in t['text']]
    assert not any('Corpus:' in text or '[old]' in text or '[new]' in text for text in texts)


def test_words_by_corpus_and_words_over_time():
    from app.ui_explore import word_change_series, words_by_corpus
    from semlens.words import usage_words
    data = word_data()
    words, order = usage_words(data, {'bubbles': 'bubble'})
    counts = words_by_corpus(data, words, order, ['old', 'new'])
    assert counts.loc['bubble'].tolist() == [3, 2] and counts.loc['balloon'].tolist() == [1, 2]
    periods = load_from_sentences(['A bubble.', 'A balloon.', 'One bubble.', 'One balloon.', 'Big bubble.', 'Red balloon.'], 'bubble',
                                  ['p1', 'p1', 'p2', 'p2', 'p3', 'p3'], forms=['balloon'])
    pwords, porder = usage_words(periods)
    array = np.eye(6, dtype=np.float32) + 0.1
    import app.ui_explore as ui
    fake = SimpleNamespace(session_state=SimpleNamespace(distance_metric='cosine', sampling_mode='balanced', sampling_repetitions=2, sampling_seed=1))
    import unittest.mock as mock
    with mock.patch.object(ui, 'st', fake):
        each, between = word_change_series(array, periods, pwords, porder, ['p1', 'p2', 'p3'], 'apd', 'previous', 'bubble')
    assert set(each) == {'bubble', 'balloon'} and len(each['bubble']) == 2 and all(np.isfinite(each['bubble']))
    assert set(between) == {'balloon'} and len(between['balloon']) == 3


def test_definitions_default_to_whole_sentences_for_different_words():
    at, data = word_app()
    widget(at, 'radio', 'Workspace').set_value('Definitions').run()
    assert not at.exception
    assert widget(at, 'selectbox', 'Input format').value == 'Definition only'
    assert widget(at, 'selectbox', 'Representation to extract').value == 'Whole input'
    at.session_state['words_mode'] = 'forms'
    at.run()
    assert widget(at, 'selectbox', 'Input format').value == 'Word: definition'


def test_the_forms_question_and_word_table_appear_in_data_and_model():
    at = AppTest.from_file('app/app.py', default_timeout=30).run()
    question = 'Are these forms of one word, or different words to compare?'
    assert not any(r.label == question for r in at.radio)
    next(t for t in at.text_input if t.label == 'Target word').set_value('bubble')
    next(t for t in at.text_input if t.label == 'Other forms (optional)').set_value('bubbles, balloon*').run()
    assert widget(at, 'radio', question).value == 'forms'
    widget(at, 'radio', question).set_value('words').run()
    assert not at.exception and at.session_state['words_mode'] == 'words'
    table = next(frame.value for frame in at.dataframe if 'Form' in frame.value.columns)
    assert list(table['Form']) == ['bubble', 'bubbles', 'balloon*'] and list(table['Word']) == ['bubble', 'bubbles', 'balloon']


def test_word_settings_are_saved_with_an_analysis(tmp_path):
    from app.ui_data import SAVED_SETTINGS
    assert {'words_mode', 'word_mapping', 'group_by', 'words_shown', 'corpora_shown'} <= set(SAVED_SETTINGS)
    data = word_data()
    embedded = embedded_usages_from_precomputed(data, torch.randn(len(data), 4))
    settings = {'words_mode': 'words', 'word_mapping': {'bubbles': 'bubble'}, 'group_by': 'word', 'words_shown': None, 'corpora_shown': ['new']}
    assert load_project(project_bytes(data, embedded, settings=settings))['settings'] == settings


def test_annotation_export_has_a_word_column_when_words_are_compared():
    from app.ui_annotations import annotation_table
    from semlens.words import usage_words
    data = word_data()
    words, _ = usage_words(data, {'bubbles': 'bubble'})
    table = annotation_table(data, {2: {'literal': None}}, {}, ['literal'], words)
    assert list(table['word'][:4]) == ['bubble', 'bubble', 'balloon', 'balloon']
    assert 'word' not in annotation_table(data, {}, {}, []).columns


def test_switching_back_and_forth_always_offers_a_valid_pair():
    at, data = word_app(group_by='word')
    widget(at, 'radio', 'Workspace').set_value('Explore').run()
    widget(at, 'selectbox', 'From word').set_value('balloon').run()
    assert widget(at, 'selectbox', 'From word').value == 'balloon'
    widget(at, 'radio', 'Colour and compare by').set_value('corpus').run()
    assert not at.exception
    assert widget(at, 'selectbox', 'From corpus').value in ('old', 'new')
    widget(at, 'radio', 'Colour and compare by').set_value('word').run()
    assert not at.exception
    assert widget(at, 'selectbox', 'From word').value in ('bubble', 'balloon')
