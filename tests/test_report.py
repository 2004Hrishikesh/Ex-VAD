"""Check that the README numbers can be reproduced from the saved scores."""

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.report import build_report, compute_metrics, load_scores, per_category_correct

ROOT = Path(__file__).resolve().parents[1]
CMAF_SCORES = ROOT / "results" / "cmaf_per_video_scores_test290.json"
TEMPORAL_SCORES = ROOT / "temporal_extension" / "results" / "per_video_scores_test290.json"


def test_cmaf_results_match_readme():
    r = build_report(CMAF_SCORES)
    assert r["exvad"]["accuracy"] == pytest.approx(0.807, abs=1e-3)
    assert r["exvad"]["recall"] == pytest.approx(0.814, abs=1e-3)
    assert r["cmaf"]["accuracy"] == pytest.approx(0.828, abs=1e-3)
    assert r["cmaf"]["recall"] == pytest.approx(0.893, abs=1e-3)
    assert r["cmaf"]["auc_roc"] == pytest.approx(0.909, abs=1e-3)
    assert r["exvad"]["fn"] - r["cmaf"]["fn"] == 11


def test_temporal_results_match_readme():
    r = build_report(TEMPORAL_SCORES)
    assert r["v5"]["accuracy"] == pytest.approx(0.862, abs=1e-3)
    assert r["v5"]["f1"] == pytest.approx(0.856, abs=1e-3)
    assert (r["cmaf"]["fp"], r["v5"]["fp"]) == (35, 19)
    assert r["v5_shuffled"]["accuracy"] < r["v5"]["accuracy"]


def test_test_split_size():
    labels, _, _ = load_scores(CMAF_SCORES)
    assert len(labels) == 290 and labels.sum() == 140


def test_compute_metrics_on_toy_example():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5
    assert m["auc_roc"] == 0.75


def test_per_category_counts():
    labels = np.array([1, 1, 0])
    cats = ["Robbery", "Robbery", "NormalVideos"]
    out = per_category_correct(labels, cats, np.array([0.9, 0.1, 0.2]), 0.5)
    assert out == {"NormalVideos": (1, 1), "Robbery": (1, 2)}


def test_report_writes_files(tmp_path):
    build_report(CMAF_SCORES, tmp_path)
    assert (tmp_path / "metrics.md").exists()
    assert (tmp_path / "roc.png").stat().st_size > 0
