"""Smoke tests: the dashboard runs on the committed results for every control setting."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'dashboard' / 'app.py')


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout = 60).run()
    assert not at.exception, at.exception
    return at


def test_default_view_shows_kpis_and_tabs(app):
    assert [m.label for m in app.metric] == [
        'Fraud value caught', 'Net saving vs no model', 'Alerts per day', 'Alert precision', 'Fraud cases caught']
    assert [t.label for t in app.tabs] == ['Operating point', 'Model performance', 'Monitoring', 'About']


@pytest.mark.parametrize('cost', [1, 5, 25, 50])
def test_review_cost_slider(app, cost):
    app.sidebar.slider[0].set_value(cost).run()
    assert not app.exception, app.exception


@pytest.mark.parametrize('rate', [1, 30, 300])
def test_custom_alert_volume(app, rate):
    app.sidebar.radio[0].set_value('Custom alert volume').run()
    app.sidebar.slider[1].set_value(rate).run()
    assert not app.exception, app.exception
