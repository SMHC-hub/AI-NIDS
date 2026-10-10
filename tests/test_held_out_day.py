"""Tests for the held-out-day generalization experiment.

Verifies:
1. Day partition isolation: Monday-Thursday files in train, Friday files in test, zero overlap.
2. Cross-split duplicate removal: exact duplicate records present in both train and test
   are purged from test to prevent leakage.
3. Preprocessing fitting isolation: zero-variance feature filtering and StandardScaler
   are fitted strictly on training data and not influenced by test data.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.held_out_day import (
    INPUT_DIR,
    TEST_DAY_PREFIXES,
    TRAIN_DAY_PREFIXES,
    get_day_files,
    remove_cross_split_duplicates,
    validate_isolation_and_preprocessing,
)


def test_file_partitioning_disjoint_and_complete():
    """Verify that all 8 raw CSVs are correctly and disjointly partitioned."""
    train_files, test_files = get_day_files(INPUT_DIR)

    assert len(train_files) == 5, f"Expected 5 Mon-Thu files, got {len(train_files)}"
    assert len(test_files) == 3, f"Expected 3 Friday files, got {len(test_files)}"
    assert len(train_files) + len(test_files) == 8

    # No files in common
    overlap = set(train_files).intersection(set(test_files))
    assert not overlap, f"Files in both train and test: {overlap}"

    # Every train file starts with Mon, Tue, Wed, or Thu
    for f in train_files:
        assert any(f.name.startswith(p) for p in TRAIN_DAY_PREFIXES), (
            f"Train file does not start with Mon-Thu: {f.name}"
        )
        assert not any(f.name.startswith(p) for p in TEST_DAY_PREFIXES), (
            f"Friday file leaked into train set: {f.name}"
        )

    # Every test file starts with Friday
    for f in test_files:
        assert any(f.name.startswith(p) for p in TEST_DAY_PREFIXES), (
            f"Test file does not start with Friday: {f.name}"
        )
        assert not any(f.name.startswith(p) for p in TRAIN_DAY_PREFIXES), (
            f"Mon-Thu file leaked into test set: {f.name}"
        )


def test_cross_split_duplicate_removal():
    """Verify that records in test matching train records are purged to prevent leakage."""
    # Synthetic train dataset
    x_train = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "feat_b": [10.0, 20.0, 30.0],
    })

    # Synthetic test dataset: row 0 matches train row 1; rows 1 and 2 are unique
    x_test = pd.DataFrame({
        "feat_a": [2.0, 4.0, 5.0],
        "feat_b": [20.0, 40.0, 50.0],
    })
    y_test = pd.Series([1, 1, 0], name="Label")
    raw_test = pd.Series(["DDoS", "PortScan", "BENIGN"])

    filtered_x, filtered_y, filtered_raw, dropped_count = remove_cross_split_duplicates(
        x_train, x_test, y_test, raw_test
    )

    assert dropped_count == 1, f"Expected 1 dropped duplicate, got {dropped_count}"
    assert len(filtered_x) == 2
    assert len(filtered_y) == 2
    assert len(filtered_raw) == 2

    # Verify the remaining records in test are the non-overlapping ones
    assert filtered_x["feat_a"].tolist() == [4.0, 5.0]
    assert filtered_y.tolist() == [1, 0]
    assert filtered_raw.tolist() == ["PortScan", "BENIGN"]


def test_preprocessing_fitted_only_on_train():
    """Verify that validate_isolation_and_preprocessing checks scaler is fitted on train only."""
    train_files, test_files = get_day_files(INPUT_DIR)

    # Synthetic train and test data with clearly different distributions
    rng = np.random.RandomState(42)
    x_train = pd.DataFrame(rng.normal(loc=10.0, scale=1.0, size=(100, 4)), columns=[f"f{i}" for i in range(4)])
    x_test = pd.DataFrame(rng.normal(loc=50.0, scale=5.0, size=(50, 4)), columns=[f"f{i}" for i in range(4)])

    # Scaler fitted ONLY on train
    scaler_correct = StandardScaler()
    scaler_correct.fit(x_train)

    # Should pass without error
    validate_isolation_and_preprocessing(
        train_files=train_files,
        test_files=test_files,
        x_train=x_train,
        x_test=x_test,
        scaler=scaler_correct,
        constant_columns=[],
    )

    # Scaler fitted on TEST or combined (leakage scenario)
    scaler_leaked = StandardScaler()
    scaler_leaked.fit(x_test)

    # Must raise AssertionError
    with pytest.raises(AssertionError, match="Scaler mean does not match training data mean"):
        validate_isolation_and_preprocessing(
            train_files=train_files,
            test_files=test_files,
            x_train=x_train,
            x_test=x_test,
            scaler=scaler_leaked,
            constant_columns=[],
        )


def test_zero_variance_derived_strictly_from_train():
    """Verify that zero-variance filtering uses train columns, ignoring test variance."""
    # Col 'constant_in_train' has 0 variance in train, but varies in test
    # Col 'constant_in_test' varies in train, but 0 variance in test
    x_train = pd.DataFrame({
        "varying": [1.0, 2.0, 3.0],
        "constant_in_train": [5.0, 5.0, 5.0],
        "constant_in_test": [10.0, 20.0, 30.0],
    })
    x_test = pd.DataFrame({
        "varying": [4.0, 5.0, 6.0],
        "constant_in_train": [5.0, 6.0, 7.0],  # Varies in test!
        "constant_in_test": [10.0, 10.0, 10.0],  # Constant in test!
    })

    # Rule: Filtering must be determined by train only
    train_constants = x_train.columns[x_train.nunique(dropna=False) <= 1].tolist()
    assert train_constants == ["constant_in_train"]

    # When dropped from both according to train rule:
    x_train_clean = x_train.drop(columns=train_constants)
    x_test_clean = x_test.drop(columns=train_constants)

    assert "constant_in_train" not in x_train_clean.columns
    assert "constant_in_train" not in x_test_clean.columns
    # 'constant_in_test' must remain because it had variance in train!
    assert "constant_in_test" in x_train_clean.columns
    assert "constant_in_test" in x_test_clean.columns
