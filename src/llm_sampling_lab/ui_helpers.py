"""Framework-independent helpers for the interactive sampling dashboard."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm_sampling_lab.sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token_indices,
)
from llm_sampling_lab.simulator import BASE_PROBABILITIES, BASE_TOKENS


def run_offline_simulation(
    strategy: str,
    value: float | int,
    samples: int,
    seed: int,
) -> list[dict[str, float | str]]:
    """Compare theoretical token probabilities with sampled frequencies."""

    if strategy not in {"temperature", "top-k", "top-p"}:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Samples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    if strategy == "temperature":
        expected = apply_temperature(BASE_PROBABILITIES, float(value))
    elif strategy == "top-k":
        expected = apply_top_k(BASE_PROBABILITIES, value)
    else:
        expected = apply_top_p(BASE_PROBABILITIES, float(value))

    draws = sample_token_indices(expected, n_samples=samples, seed=seed)
    counts = np.bincount(draws, minlength=len(BASE_TOKENS))
    observed = counts / samples

    return [
        {
            "Token": token,
            "Expected": float(expected[index]),
            "Observed": float(observed[index]),
        }
        for index, token in enumerate(BASE_TOKENS)
    ]


def list_report_files(
    directory: Path,
    limit: int = 100,
) -> list[Path]:
    """Find recent experiment JSON reports within a trusted local directory."""

    if limit < 1:
        raise ValueError("limit must be positive.")
    if not directory.exists():
        return []
    files = [path for path in directory.rglob("*.json") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def load_report(path: Path) -> dict[str, Any]:
    """Read one persisted JSON report."""

    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("Report must be a JSON object.")
    return report
