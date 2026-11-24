"""Settings remembered between sessions."""
import json

from streamlit.testing.v1 import AppTest

import app.preferences as preferences


def test_settings_are_remembered_for_the_next_session():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    at.sidebar.slider(key='point_size').set_value(14).run()
    at.sidebar.radio(key='theme_mode').set_value('Dark').run()
    at.number_input(key='encoder_layer').set_value(8).run()
    saved = json.loads(preferences.PREFERENCES_FILE.read_text())
    assert saved['point_size'] == 14 and saved['theme_mode'] == 'Dark' and saved['encoder_layer'] == 8
    assert 'target_word' not in saved and 'source_texts' not in saved
    # Settings left at their defaults are not written, so later versions can improve them.
    assert 'distance_metric' not in saved and 'dv_cap' not in saved
    later = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert later.session_state['point_size'] == 14 and later.session_state['theme_mode'] == 'Dark'
    assert later.number_input(key='encoder_layer').value == 8 and not later.exception and not later.warning


def test_embedding_settings_survive_other_workspaces():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    at.number_input(key='encoder_layer').set_value(-2).run()
    at.number_input(key='encoder_batch_size').set_value(4).run()
    at.button_group(key='workspace').set_value('Explore').run()
    at.button_group(key='workspace').set_value('Data & Model').run()
    assert at.number_input(key='encoder_layer').value == -2 and at.number_input(key='encoder_batch_size').value == 4


def test_unusable_saved_values_are_ignored_and_defaults_can_be_restored():
    preferences.PREFERENCES_FILE.write_text(json.dumps({'point_size': 400, 'theme_mode': 'Purple', 'palette_name': 'No such palette',
                                                        'figure_format': 'svg', 'target_word': 'bubble', 'dv_layer_x': 6}))
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    assert at.session_state['point_size'] == 9 and at.session_state['theme_mode'] == 'Light'
    assert at.session_state['figure_format'] == 'svg' and at.session_state['target_word'] == '' and at.session_state['dv_layer_x'] == 6
    next(b for b in at.sidebar.button if b.label == 'Restore the default settings').click().run()
    assert at.session_state['figure_format'] == 'png' and 'dv_layer_x' not in at.session_state
    assert not at.exception
    assert not preferences.PREFERENCES_FILE.exists() or json.loads(preferences.PREFERENCES_FILE.read_text()) == {}
    preferences.PREFERENCES_FILE.write_text('not json')
    assert preferences.read_preferences() == {}


def test_page_tabs_keep_the_open_page_when_it_is_clicked_again():
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    at.button_group(key='workspace').set_value('Definitions').run()
    at.button_group(key='workspace').set_value(None).run()
    assert at.session_state['workspace'] == 'Definitions' and not at.exception
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    at.button_group(key='dv_step').set_value('Seeds').run()
    at.button_group(key='dv_step').set_value(None).run()
    assert at.session_state['dv_step'] == 'Seeds' and at.text_input(key='dv_seed_words_input') is not None
