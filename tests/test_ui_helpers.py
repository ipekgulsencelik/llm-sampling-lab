"""Tests for the web UI's framework-independent behavior."""

import json

import pytest

from llm_sampling_lab.ui_helpers import (
    list_report_files,
    load_report,
    run_offline_simulation,
)


@pytest.mark.parametrize(
    ("strategy", "value"),
    [
        ("temperature", 0.5),
        ("top-k", 3),
        ("top-p", 0.7),
    ],
)
def test_offline_simulation_is_reproducible(strategy, value):
    first = run_offline_simulation(strategy, value, samples=1000, seed=42)
    second = run_offline_simulation(strategy, value, samples=1000, seed=42)

    assert first == second
    assert len(first) == 5
    assert sum(row["Expected"] for row in first) == pytest.approx(1.0)
    assert sum(row["Observed"] for row in first) == pytest.approx(1.0)


def test_top_k_removes_unselected_tokens():
    rows = run_offline_simulation("top-k", 1, samples=200, seed=42)
    assert rows[0]["Expected"] == pytest.approx(1.0)
    assert all(row["Expected"] == 0 for row in rows[1:])


@pytest.mark.parametrize(
    ("strategy", "value", "samples", "seed"),
    [
        ("invalid", 1.0, 100, 42),
        ("temperature", 1.0, 0, 42),
        ("temperature", 1.0, 100, -1),
        ("top-k", 7, 100, 42),
    ],
)
def test_invalid_offline_inputs_raise(strategy, value, samples, seed):
    with pytest.raises(ValueError):
        run_offline_simulation(strategy, value, samples, seed)


def test_report_listing_and_loading(tmp_path):
    assert list_report_files(tmp_path) == []
    path = tmp_path / "results" / "temperature" / "example.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert list_report_files(tmp_path / "results") == [path]
    assert load_report(path)["status"] == "completed"
