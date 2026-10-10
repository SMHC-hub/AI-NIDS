"""Lightweight smoke tests for the trained AI-NIDS Random Forest."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xai_engine import explain_one, generate_soc_report

MODEL_PATH = ROOT / "models" / "rf_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
FEATURES_PATH = ROOT / "models" / "feature_columns.pkl"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


@pytest.fixture(scope="module")
def sample(artifacts):
    _, _, features = artifacts
    frame = pd.read_csv(TEST_PATH, nrows=10)
    missing = [name for name in features if name not in frame.columns]
    assert not missing, f"Missing feature columns: {missing}"
    return frame.loc[:, features].to_numpy(dtype=np.float64)


def test_artifacts_exist():
    for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH, TEST_PATH):
        assert path.is_file(), f"Missing artifact: {path}"


def test_feature_compatibility(artifacts):
    model, scaler, features = artifacts
    assert len(features) == 70
    assert len(set(features)) == len(features)
    assert model.n_features_in_ == scaler.n_features_in_ == len(features)
    assert list(model.classes_) == [0, 1]


def test_predictions_are_binary(artifacts, sample):
    model, _, _ = artifacts
    predictions = model.predict(sample)
    assert predictions.shape == (len(sample),)
    assert set(predictions).issubset({0, 1})


def test_probabilities_are_valid(artifacts, sample):
    model, _, _ = artifacts
    probabilities = model.predict_proba(sample)
    assert probabilities.shape == (len(sample), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_array_equal(model.predict(sample), model.classes_[np.argmax(probabilities, axis=1)])

def test_shap_explanation(artifacts, sample):
    model, _, features = artifacts

    raw_frame = pd.read_csv(
        ROOT / "data" / "processed" / "test_raw.csv",
        nrows=1
    )

    raw_row = raw_frame.iloc[0]

    predicted, probability, details = explain_one(
        model=model,
        features=features,
        scaled_row=sample[0],
        raw_row=raw_row,
        top_k=5
    )

    assert predicted in (0, 1)
    assert 0 <= probability <= 1
    assert len(details) == 5

    for item in details:
        assert item["feature"] in features
        assert np.isfinite(item["shap_value"])
        assert item["direction"] in (
            "supports predicted class",
            "opposes predicted class"
        )

def test_groq_missing_api_key(monkeypatch):
    # Simulate an environment without a Groq API key.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Prevent the function from loading the real .env file.
    monkeypatch.setattr("xai_engine.load_dotenv", lambda *args, **kwargs: None)

    report = generate_soc_report(
        predicted=1,
        probability=0.95,
        details=[]
    )

    assert isinstance(report, str)
    assert "Groq unavailable" in report
    assert "Local SHAP results remain available" in report

def test_groq_api_failure(monkeypatch):
    import groq

    # Use a temporary fake API key.
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    # Prevent loading the real .env file.
    monkeypatch.setattr(
        "xai_engine.load_dotenv",
        lambda *args, **kwargs: None
    )

    # Simulate an unavailable Groq service.
    class FakeGroq:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("Simulated API failure")

    monkeypatch.setattr(groq, "Groq", FakeGroq)

    report = generate_soc_report(
        predicted=1,
        probability=0.95,
        details=[]
    )

    assert isinstance(report, str)
    assert "Groq analysis unavailable" in report
    assert "ConnectionError" in report
    assert "Local SHAP results remain available" in report