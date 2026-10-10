
"""Phase 1: CIC-IDS2017 preprocessing for the AI-NIDS baseline.

Run from the project root: python preprocess.py
Raw files are read only; artifacts are written to data/processed and models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "CIC-IDS-2017"
    / "CSVs"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
)

OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

CHUNK_SIZE = 100_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger("ai_nids.preprocess")


def load_dataset() -> pd.DataFrame:
    """Load and combine the eight CIC-IDS2017 CSV files."""

    files = sorted(INPUT_DIR.glob("*.csv"))

    if len(files) != 8:
        raise FileNotFoundError(
            f"Expected 8 CSV files in {INPUT_DIR}; found {len(files)}"
        )

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None

    for file in files:
        logger.info("Reading %s", file.name)

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8-sig"
        ):
            chunk.columns = chunk.columns.str.strip()

            if chunk.columns.duplicated().any():
                raise ValueError(
                    f"Duplicate column names in {file.name}"
                )

            if expected_columns is None:
                expected_columns = list(chunk.columns)

                if "Label" not in expected_columns:
                    raise ValueError("Label column missing")

            elif list(chunk.columns) != expected_columns:
                raise ValueError(
                    f"Column mismatch in {file.name}"
                )

            frames.append(chunk)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Clean, split, scale, and save the CIC-IDS2017 dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: LOAD DATASET
    # ---------------------------------------------------------

    df = load_dataset()

    logger.info(
        "Loaded %s rows and %s columns",
        f"{len(df):,}",
        len(df.columns)
    )

    # ---------------------------------------------------------
    # STEP 2: REMOVE DUPLICATES
    # ---------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates().copy()

    logger.info(
        "Removed %s duplicate rows",
        f"{before - len(df):,}"
    )

    # ---------------------------------------------------------
    # STEP 3: PREPARE BINARY LABELS
    # ---------------------------------------------------------

    labels = (
        df.pop("Label")
        .astype("string")
        .str.strip()
        .str.upper()
    )

    invalid_labels = labels.isna() | labels.eq("")

    if invalid_labels.any():
        logger.warning(
            "Dropping %s rows with missing labels",
            f"{invalid_labels.sum():,}"
        )

        df = df.loc[~invalid_labels].copy()
        labels = labels.loc[~invalid_labels]

    # 0 = BENIGN
    # 1 = MALICIOUS
    target = labels.ne("BENIGN").astype("int8")

    # ---------------------------------------------------------
    # STEP 4: CLEAN NUMERICAL FEATURES
    # ---------------------------------------------------------

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = df.notna().all(axis=1)

    logger.info(
        "Dropping %s rows with missing/non-finite features",
        f"{(~valid).sum():,}"
    )

    df = df.loc[valid].copy()
    target = target.loc[valid].copy()

    if df.empty or target.nunique() != 2:
        raise ValueError(
            "Dataset must have nonempty features and both binary classes"
        )

    # ---------------------------------------------------------
    # STEP 5: SPLIT DATASET BEFORE FEATURE SELECTION
    # ---------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        df,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target
    )

    logger.info(
        "Training rows: %s",
        f"{len(x_train):,}"
    )

    logger.info(
        "Testing rows: %s",
        f"{len(x_test):,}"
    )

    # ---------------------------------------------------------
    # STEP 6: REMOVE ZERO-VARIANCE FEATURES
    # Training data determines which columns to remove.
    # ---------------------------------------------------------

    constant_columns = x_train.columns[
        x_train.nunique(dropna=False) <= 1
    ].tolist()

    x_train = x_train.drop(
        columns=constant_columns
    )

    x_test = x_test.drop(
        columns=constant_columns
    )

    logger.info(
        "Removed %s zero-variance features: %s",
        len(constant_columns),
        constant_columns
    )

    if x_train.shape[1] == 0:
        raise ValueError(
            "No usable features remain after filtering"
        )

    # ---------------------------------------------------------
    # STEP 7: SCALE FEATURES
    # Fit on training data only.
    # ---------------------------------------------------------

    feature_columns = x_train.columns.tolist()

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)

    logger.info(
        "Retained %s features",
        len(feature_columns)
    )

    # ---------------------------------------------------------
    # STEP 8: SAVE PROCESSED TRAIN AND TEST DATASETS
    # ---------------------------------------------------------

    for filename, features, labels_out in [
        ("train.csv", train_scaled, y_train),
        ("test.csv", test_scaled, y_test),
    ]:

        output = pd.DataFrame(
            features,
            columns=feature_columns
        )

        output["Label"] = labels_out.to_numpy(
            dtype=np.int8
        )

        output.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

        logger.info(
            "Saved %s (%s rows)",
            filename,
            f"{len(output):,}"
        )

    # ---------------------------------------------------------
    # STEP 9: SAVE UNSCALED TEST DATA
    # Required for displaying original feature values in SHAP.
    # ---------------------------------------------------------

    raw_test = x_test.reset_index(
        drop=True
    ).copy()

    raw_test["Label"] = y_test.to_numpy(
        dtype=np.int8
    )

    raw_test.to_csv(
        OUTPUT_DIR / "test_raw.csv",
        index=False
    )

    logger.info("Saved test_raw.csv")

    # ---------------------------------------------------------
    # STEP 10: SAVE SCALER AND FEATURE COLUMNS
    # ---------------------------------------------------------

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    joblib.dump(
        feature_columns,
        MODEL_DIR / "feature_columns.pkl"
    )

    logger.info("Saved scaler and feature columns")

    # ---------------------------------------------------------
    # STEP 11: SAVE PREPROCESSING SUMMARY
    # ---------------------------------------------------------

    summary = {
        "source_files": len(list(INPUT_DIR.glob("*.csv"))),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": len(feature_columns),
        "removed_zero_variance_features": constant_columns,
        "train_class_counts": {
            str(k): int(v)
            for k, v in y_train.value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v)
            for k, v in y_test.value_counts().items()
        },
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_SIZE,
    }

    summary_path = (
        OUTPUT_DIR / "preprocessing_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    logger.info(
        "Phase 1 complete. %s features retained.",
        len(feature_columns)
    )


if __name__ == "__main__":
    main()
