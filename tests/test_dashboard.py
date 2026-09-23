from pathlib import Path

from streamlit.testing.v1 import AppTest


def dashboard_app() -> AppTest:
    path = Path(__file__).parents[1] / "app" / "dashboard.py"
    return AppTest.from_file(str(path), default_timeout=20)


def test_agent_tab_renders_and_runs_methodology_workflow():
    app = dashboard_app().run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Single record",
        "CSV batch",
        "Agent assistant",
    ]

    app.radio[0].set_value("Ask about methodology").run()
    app.button[-1].click().run()

    assert not app.exception
    assert "docs/MODEL_CARD.md" in app.success[-1].value
    assert "not medical advice" in app.info[-1].value


def test_natural_language_agent_workflow():
    app = dashboard_app().run()
    app.text_area[0].set_value("Is profile ka risk score explain karo").run()
    app.button[-1].click().run()

    assert not app.exception
    assert "synthetic demo score" in app.success[-1].value.lower()
    assert any("Selected intent" in markdown.value for markdown in app.markdown)
