"""Shared test set-up."""
import pytest


@pytest.fixture(autouse=True)
def separate_settings(tmp_path_factory, monkeypatch):
    """Tests never read or write the remembered settings of the person running them."""
    import app.preferences as preferences
    monkeypatch.setattr(preferences, 'PREFERENCES_FILE', tmp_path_factory.mktemp('settings') / 'settings.json')
