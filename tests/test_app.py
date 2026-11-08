"""The Streamlit app runs headless, end to end, on the char model."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


def test_app_runs_without_exceptions() -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    assert [t.label for t in at.tabs] == [
        "Distribution viewer",
        "Decoder race",
        "Temperature sweep",
        "Task matcher",
    ]


def test_app_task_matcher_switches_to_open_ended() -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    task_radio = next(r for r in at.radio if r.label == "Task")
    task_radio.set_value("open").run()
    assert not at.exception, at.exception
    assert any("Arena winner" in s.value for s in at.success)
