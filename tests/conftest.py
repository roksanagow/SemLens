"""Shared test set-up."""
import json
import pytest


@pytest.fixture(autouse=True)
def separate_settings(tmp_path_factory, monkeypatch):
    """Tests never read or write the settings, kept work or saved files of the person running them.

    The start page has been seen, as for anyone after their first visit, so that
    tests open on the workspaces; the start page's own tests remove the file.
    Advanced options are on, so that every setting can be reached; the tests of
    the switch itself turn it off.
    """
    import app.preferences as preferences
    import app.autosave as autosave
    settings = tmp_path_factory.mktemp('settings') / 'settings.json'
    settings.write_text(json.dumps({'start_seen': True, 'advanced_options': True}))
    monkeypatch.setattr(preferences, 'PREFERENCES_FILE', settings)
    monkeypatch.setattr(autosave, 'AUTOSAVE_ROOT', tmp_path_factory.mktemp('autosave'))
    # Saved analyses and discoveries: a test sees only those it makes itself.
    import app.ui_data as ui_data
    import app.discover as discover
    storage = tmp_path_factory.mktemp('storage')
    monkeypatch.setattr(ui_data, 'STORAGE', storage)
    monkeypatch.setattr(discover, 'DISCOVER_ROOT', storage / 'discover')
