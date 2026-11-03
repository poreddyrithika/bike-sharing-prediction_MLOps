"""Unit tests that need no MLflow state and no deployment bundle."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "mlops_pipeline"))

from validate_data import validate_schema  # noqa: E402
import common  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "src" / "mlops_pipeline" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw():
    return pd.read_csv(ROOT / "data" / "raw" / "day.csv")


def test_raw_data_passes_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert validate_schema(raw())


def test_schema_catches_each_corruption(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = {
        "range": lambda d: d.assign(hum=d.hum.where(d.index != 0, 1.9)),
        "category": lambda d: d.assign(season=d.season.where(d.index != 0, 7)),
        "negative": lambda d: d.assign(cnt=d.cnt.where(d.index != 0, -5)),
        "missing_col": lambda d: d.drop(columns=["weathersit"]),
        "extra_col": lambda d: d.assign(extra=1),
        "inconsistent": lambda d: d.assign(registered=d.registered.where(d.index != 0, 9999)),
        "dup_instant": lambda d: d.assign(instant=d.instant.where(d.index != 1, 1)),
    }
    for name, mutate in cases.items():
        assert not validate_schema(mutate(raw())), name


def test_clean_dataframe_drops_leakage_and_ids():
    config = common.load_config()
    cleaned = common.clean_dataframe(raw(), config)
    assert list(cleaned.columns) == common.feature_columns(config)
    assert not {"cnt", "casual", "registered", "instant", "dteday"} & set(cleaned.columns)


def test_preprocessor_has_stable_width_even_for_unseen_year():
    config = common.load_config()
    df = raw()
    pre = common.build_preprocessor(config).fit(common.clean_dataframe(df.head(300), config))
    df2 = df.head(5).assign(yr=2)
    out = pre.transform(common.clean_dataframe(df2, config))
    assert out.shape[1] == common.expected_transformed_width(config)


def test_js_divergence_bounds():
    drift = _load("12_data_drift")
    same = pd.Series([1, 1, 2, 3])
    assert drift.js_divergence(same, same) == 0
    assert 0.99 < drift.js_divergence(pd.Series([1, 1]), pd.Series([2, 2])) <= 1.0


def test_metrics_perfect_prediction():
    m = common.regression_metrics(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))
    assert m["rmse"] == 0 and m["r2_score"] == 1
