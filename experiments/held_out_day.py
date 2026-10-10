"""Held-Out-Day Generalization Experiment for AI-NIDS.

Objective:
    Train a Random Forest network intrusion detection model using CIC-IDS2017
    traffic from Monday through Thursday and evaluate it strictly on Friday traffic.

Context & Research Purpose:
    In standard flow-based intrusion detection evaluations, random stratified
    splitting mixes flows from the same capture sessions across train and test splits,
    which can artificially inflate evaluation scores.

    Evaluating across temporal capture days (held-out-day generalization) evaluates
    the model against distribution shifts. Crucially, in CIC-IDS2017:
      - Monday-Thursday contains: BENIGN, DoS (Hulk, GoldenEye, slowloris, Slowhttptest),
        Heartbleed, Web Attacks (Brute Force, XSS, SQL Injection), Infiltration,
        and Patator attacks (FTP, SSH).
      - Friday contains: BENIGN, DDoS, PortScan, and Bot.

    All malicious attack categories on Friday (DDoS, PortScan, Botnet) were NOT present
    during Monday–Thursday training. Consequently, this experiment evaluates
    *unseen-attack zero-day / out-of-distribution generalization*, not standard
    in-distribution classification.

Artifacts and Isolation:
    All experiment outputs are strictly written to `experiments/results/held_out_day/`.
    The existing baseline artifacts in `models/` and `data/processed/` are NEVER overwritten.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "CIC-IDS-2017"
    / "CSVs"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
)
RESULTS_DIR = ROOT / "experiments" / "results" / "held_out_day"

CHUNK_SIZE = 100_000
RANDOM_STATE = 42

# Baseline Random Forest hyperparameters (identical to train_model.py)
RF_HYPERPARAMETERS = {
    "n_estimators": 80,
    "max_depth": 16,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": 4,
    "random_state": RANDOM_STATE,
}

TRAIN_DAY_PREFIXES = ("Monday", "Tuesday", "Wednesday", "Thursday")
TEST_DAY_PREFIXES = ("Friday",)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_nids.held_out_day")


def get_day_files(input_dir: Path = INPUT_DIR) -> tuple[list[Path], list[Path]]:
    """Partition the eight CIC-IDS2017 CSV files into Monday-Thursday (train) and Friday (test)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {input_dir}")

    all_csvs = sorted(input_dir.glob("*.csv"))
    if len(all_csvs) != 8:
        raise FileNotFoundError(
            f"Expected exactly 8 CSV files in {input_dir}; found {len(all_csvs)}"
        )

    train_files: list[Path] = []
    test_files: list[Path] = []

    for path in all_csvs:
        filename = path.name
        if any(filename.startswith(prefix) for prefix in TRAIN_DAY_PREFIXES):
            train_files.append(path)
        elif any(filename.startswith(prefix) for prefix in TEST_DAY_PREFIXES):
            test_files.append(path)
        else:
            raise ValueError(f"Unrecognized file prefix for day partition: {filename}")

    if len(train_files) != 5:
        raise ValueError(
            f"Expected 5 Monday-Thursday training files, found {len(train_files)}: "
            f"{[f.name for f in train_files]}"
        )
    if len(test_files) != 3:
        raise ValueError(
            f"Expected 3 Friday testing files, found {len(test_files)}: "
            f"{[f.name for f in test_files]}"
        )

    # Invariant: train and test files must be strictly disjoint
    if set(train_files).intersection(set(test_files)):
        raise RuntimeError("Data leakage detected: train and test file sets overlap!")

    return train_files, test_files


def load_and_clean_day_split(
    files: list[Path],
    split_name: str,
    max_rows_per_file: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Load, combine, clean, and deduplicate CSV files for a day partition.

    Returns:
        features_df: Cleaned float32 feature columns.
        binary_target: pd.Series of int8 (0 = BENIGN, 1 = MALICIOUS).
        raw_labels: pd.Series of string original attack names.
        stats: Dictionary of filtering and cleaning metrics.
    """
    logger.info("Loading %s partition (%d files)...", split_name, len(files))

    expected_columns: list[str] | None = None
    frames: list[pd.DataFrame] = []
    raw_rows_total = 0

    for file in files:
        logger.info("  Reading %s", file.name)
        file_rows = 0

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8-sig",
        ):
            chunk.columns = chunk.columns.str.strip()

            if chunk.columns.duplicated().any():
                raise ValueError(f"Duplicate columns in {file.name}")

            if expected_columns is None:
                expected_columns = list(chunk.columns)
                if "Label" not in expected_columns:
                    raise ValueError(f"'Label' column missing in {file.name}")
            elif list(chunk.columns) != expected_columns:
                raise ValueError(f"Column schema mismatch in {file.name}")

            raw_rows_total += len(chunk)
            file_rows += len(chunk)
            frames.append(chunk)

            if max_rows_per_file and file_rows >= max_rows_per_file:
                logger.info("  Reached max_rows_per_file limit (%d) for %s", max_rows_per_file, file.name)
                break

    # Concatenate raw chunks
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    initial_rows = len(df)
    logger.info("  Total rows loaded for %s: %s", split_name, f"{initial_rows:,}")

    # Step 1: Internal deduplication within the split (matching preprocess.py)
    before_dedup = len(df)
    df = df.drop_duplicates().copy()
    internal_duplicates_dropped = before_dedup - len(df)
    logger.info("  Dropped %s internal duplicate rows in %s", f"{internal_duplicates_dropped:,}", split_name)

    # Step 2: Separate labels and prepare binary target
    raw_labels = df.pop("Label").astype("string").str.strip()
    invalid_labels = raw_labels.isna() | raw_labels.eq("")
    invalid_label_count = int(invalid_labels.sum())
    if invalid_label_count > 0:
        logger.warning("  Dropping %d rows with missing labels in %s", invalid_label_count, split_name)
        df = df.loc[~invalid_labels].copy()
        raw_labels = raw_labels.loc[~invalid_labels].copy()

    # 0 = BENIGN, 1 = MALICIOUS
    binary_target = raw_labels.str.upper().ne("BENIGN").astype(np.int8)

    # Step 3: Numerical cleaning (convert to numeric, replace inf/-inf with nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    valid_features_mask = df.notna().all(axis=1)
    non_finite_dropped = int((~valid_features_mask).sum())

    features_df = df.loc[valid_features_mask].astype(np.float32).reset_index(drop=True)
    binary_target = binary_target.loc[valid_features_mask].reset_index(drop=True)
    raw_labels = raw_labels.loc[valid_features_mask].reset_index(drop=True)
    del df
    gc.collect()

    logger.info(
        "  Dropped %s rows with missing/non-finite features in %s",
        f"{non_finite_dropped:,}",
        split_name,
    )

    stats = {
        "split_name": split_name,
        "files_count": len(files),
        "raw_rows_loaded": initial_rows,
        "internal_duplicates_dropped": internal_duplicates_dropped,
        "invalid_labels_dropped": invalid_label_count,
        "non_finite_dropped": non_finite_dropped,
        "clean_rows": len(features_df),
        "class_distribution": {
            "BENIGN_0": int((binary_target == 0).sum()),
            "MALICIOUS_1": int((binary_target == 1).sum()),
        },
        "attack_breakdown": {
            str(k): int(v) for k, v in raw_labels.value_counts().items()
        },
    }

    return features_df, binary_target, raw_labels, stats


def remove_cross_split_duplicates(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    raw_test: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    """Prevent cross-split leakage by removing records in test that exactly match any record in train.

    Uses 64-bit row hashing with targeted candidate verification to minimize memory usage.
    Returns:
        Filtered (x_test, y_test, raw_test) and count of records dropped from test.
    """
    logger.info("Checking for cross-split duplicate leakage between Train and Friday Test...")

    # Compute vectorized 64-bit row hashes
    train_hash_series = pd.util.hash_pandas_object(x_train, index=False)
    train_hashes = set(train_hash_series)
    test_hash_series = pd.util.hash_pandas_object(x_test, index=False)

    candidate_mask = test_hash_series.isin(train_hashes)
    candidate_indices = np.where(candidate_mask)[0]

    logger.info("  Identified %d hash-match candidate rows in test", len(candidate_indices))

    if len(candidate_indices) == 0:
        logger.info("  Zero cross-split duplicate records found.")
        return x_test, y_test, raw_test, 0

    # Narrow down train rows to ONLY those matching candidate hashes (avoids merging millions of rows)
    matched_candidate_hashes = set(test_hash_series.iloc[candidate_indices])
    train_candidate_mask = train_hash_series.isin(matched_candidate_hashes)
    train_subset = x_train.loc[train_candidate_mask]

    test_candidates = x_test.iloc[candidate_indices]
    merged = test_candidates.reset_index().merge(
        train_subset,
        on=list(x_train.columns),
        how="inner",
    )
    confirmed_leak_indices = merged["index"].unique()
    leak_count = len(confirmed_leak_indices)

    logger.info(
        "  Confirmed %d exact cross-split duplicate records in Friday test. Dropping to prevent leakage.",
        leak_count,
    )

    keep_mask = ~x_test.index.isin(confirmed_leak_indices)
    x_test_filtered = x_test.loc[keep_mask].reset_index(drop=True)
    y_test_filtered = y_test.loc[keep_mask].reset_index(drop=True)
    raw_test_filtered = raw_test.loc[keep_mask].reset_index(drop=True)

    del train_hash_series, train_hashes, test_hash_series, candidate_mask, test_candidates, train_subset, merged
    gc.collect()

    return x_test_filtered, y_test_filtered, raw_test_filtered, leak_count


def validate_isolation_and_preprocessing(
    train_files: list[Path],
    test_files: list[Path],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    scaler: StandardScaler,
    constant_columns: list[str],
) -> None:
    """Rigorous sanity checks verifying complete train/test isolation and proper preprocessing fitting."""
    logger.info("Validating isolation and preprocessing invariants...")

    # Invariant 1: File sets are strictly partitioned
    for f in test_files:
        assert any(f.name.startswith(p) for p in TEST_DAY_PREFIXES), (
            f"Non-Friday file in test set: {f.name}"
        )
    for f in train_files:
        assert any(f.name.startswith(p) for p in TRAIN_DAY_PREFIXES), (
            f"Friday file in train set: {f.name}"
        )
    assert set(train_files).isdisjoint(set(test_files)), "Train and test files overlap!"

    # Invariant 2: Feature alignment
    assert x_train.columns.tolist() == x_test.columns.tolist(), "Feature column mismatch between train and test!"
    for col in constant_columns:
        assert col not in x_train.columns, f"Zero-variance feature {col} not dropped from train!"
        assert col not in x_test.columns, f"Zero-variance feature {col} not dropped from test!"

    # Invariant 3: Scaler was fitted strictly on train data
    assert scaler.n_features_in_ == x_train.shape[1], (
        f"Scaler features ({scaler.n_features_in_}) != train features ({x_train.shape[1]})"
    )
    # Check that scaler mean matches train mean within numerical precision
    train_means = x_train.mean(axis=0).to_numpy()
    np.testing.assert_allclose(
        scaler.mean_,
        train_means,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Scaler mean does not match training data mean! Preprocessing was not fit on train.",
    )

    logger.info("Validation successful: complete day isolation and preprocessing integrity confirmed.")


def run_experiment(
    max_rows_per_file: int | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Execute the held-out-day generalization experiment."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Identify file partitions
    train_files, test_files = get_day_files(INPUT_DIR)
    logger.info(
        "Files partition:\n  Train (Mon-Thu, %d files): %s\n  Test (Fri, %d files): %s",
        len(train_files),
        [f.name for f in train_files],
        len(test_files),
        [f.name for f in test_files],
    )

    # Step 2: Load and clean training partition (Monday-Thursday)
    x_train, y_train, raw_train, train_stats = load_and_clean_day_split(
        train_files, "Train (Monday-Thursday)", max_rows_per_file=max_rows_per_file
    )

    # Step 3: Load and clean testing partition (Friday)
    x_test, y_test, raw_test, test_stats = load_and_clean_day_split(
        test_files, "Test (Friday)", max_rows_per_file=max_rows_per_file
    )

    # Step 4: Remove cross-split duplicate records from Friday test to prevent leakage
    x_test, y_test, raw_test, cross_leak_dropped = remove_cross_split_duplicates(
        x_train, x_test, y_test, raw_test
    )
    test_stats["cross_split_leakage_dropped"] = cross_leak_dropped
    test_stats["final_evaluated_rows"] = len(x_test)

    # Step 5: Filter zero-variance features based STRICTLY on training data
    constant_columns = x_train.columns[
        x_train.nunique(dropna=False) <= 1
    ].tolist()
    logger.info(
        "Identified %d zero-variance features in training data: %s",
        len(constant_columns),
        constant_columns,
    )

    x_train = x_train.drop(columns=constant_columns)
    x_test = x_test.drop(columns=constant_columns)
    feature_columns = x_train.columns.tolist()
    logger.info("Retained %d features for modeling", len(feature_columns))

    # Step 6: Fit scaler on training data only, apply to both splits
    scaler = StandardScaler()
    logger.info("Fitting StandardScaler on training data only...")
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    # Step 7: Sanity check isolation & fitting
    validate_isolation_and_preprocessing(
        train_files, test_files, x_train, x_test, scaler, constant_columns
    )

    # Step 8: Train Random Forest baseline
    logger.info(
        "Initializing Random Forest classifier (same hyperparameters as baseline: %s)...",
        RF_HYPERPARAMETERS,
    )
    model = RandomForestClassifier(**RF_HYPERPARAMETERS)

    logger.info("Training Random Forest on Monday-Thursday data (%s rows)...", f"{len(x_train_scaled):,}")
    t_start = time.perf_counter()
    model.fit(x_train_scaled, y_train.to_numpy(dtype=np.int8))
    train_duration = time.perf_counter() - t_start
    logger.info("Training complete in %.2f seconds (%.2f minutes)", train_duration, train_duration / 60)

    # Step 9: Evaluate on held-out Friday test data
    logger.info("Evaluating on held-out Friday data (%s rows)...", f"{len(x_test_scaled):,}")
    y_pred = model.predict(x_test_scaled)

    # Overall binary metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    logger.info("=== HELD-OUT FRIDAY GENERALIZATION METRICS ===")
    logger.info("Accuracy:             %.6f", acc)
    logger.info("Precision (Malicious): %.6f", prec)
    logger.info("Recall (Malicious):    %.6f", rec)
    logger.info("F1 Score (Malicious):  %.6f", f1)
    logger.info("Confusion Matrix [[TN, FP], [FN, TP]]:\n%s", cm)

    # Step 10: Compute per-attack detection recall (identifying seen vs. unseen attacks)
    train_attack_categories = set(raw_train.unique())
    test_eval_df = pd.DataFrame({
        "raw_label": raw_test,
        "true_binary": y_test,
        "predicted_binary": y_pred,
    })

    per_attack_records: list[dict[str, Any]] = []
    for label, group in test_eval_df.groupby("raw_label"):
        support = len(group)
        is_benign = str(label).upper() == "BENIGN"
        seen_in_train = str(label) in train_attack_categories

        if is_benign:
            correct = int((group["predicted_binary"] == 0).sum())
            fp_count = int((group["predicted_binary"] == 1).sum())
            recall_or_specificity = correct / support if support > 0 else 0.0
            per_attack_records.append({
                "label": str(label),
                "category_type": "BENIGN",
                "seen_in_training": True,
                "support": support,
                "detected_malicious": fp_count,
                "correct_classifications": correct,
                "metric_name": "Specificity (Benign Accuracy)",
                "metric_value": round(recall_or_specificity, 6),
                "false_positive_rate": round(fp_count / support, 6) if support > 0 else 0.0,
            })
        else:
            detected = int((group["predicted_binary"] == 1).sum())
            fn_count = int((group["predicted_binary"] == 0).sum())
            attack_recall = detected / support if support > 0 else 0.0
            per_attack_records.append({
                "label": str(label),
                "category_type": "MALICIOUS",
                "seen_in_training": seen_in_train,
                "support": support,
                "detected_malicious": detected,
                "missed_malicious_fn": fn_count,
                "metric_name": "Detection Recall",
                "metric_value": round(attack_recall, 6),
                "false_positive_rate": 0.0,
            })

    per_attack_df = pd.DataFrame(per_attack_records)
    print("\n=== PER-ATTACK DETECTION BREAKDOWN (FRIDAY TEST) ===")
    print(per_attack_df.to_string(index=False))

    # Step 11: Feature importances
    feature_importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Step 12: Assemble comprehensive results payload
    unseen_attacks_list = [
        r["label"] for r in per_attack_records
        if r["category_type"] == "MALICIOUS" and not r["seen_in_training"]
    ]

    experiment_summary = {
        "experiment_name": "held_out_day_generalization",
        "description": (
            "Temporal generalization experiment: Trained on Monday-Thursday traffic, "
            "evaluated strictly on Friday traffic. Tests out-of-distribution detection on "
            "unseen attack families."
        ),
        "train_days": list(TRAIN_DAY_PREFIXES),
        "test_days": list(TEST_DAY_PREFIXES),
        "train_files": [f.name for f in train_files],
        "test_files": [f.name for f in test_files],
        "unseen_attack_categories": unseen_attacks_list,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "feature_count": len(feature_columns),
        "removed_zero_variance_features": constant_columns,
        "rf_hyperparameters": RF_HYPERPARAMETERS,
        "training_duration_seconds": round(train_duration, 2),
        "overall_metrics": {
            "accuracy": acc,
            "precision_malicious": prec,
            "recall_malicious": rec,
            "f1_malicious": f1,
            "confusion_matrix": cm.tolist(),
        },
        "per_attack_metrics": per_attack_records,
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["BENIGN", "MALICIOUS"], zero_division=0, output_dict=True
        ),
    }

    # Step 13: Save all artifacts strictly in `experiments/results/held_out_day/`
    logger.info("Saving experiment artifacts to %s...", results_dir)

    metrics_file = results_dir / "held_out_day_metrics.json"
    metrics_file.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")

    per_attack_df.to_csv(results_dir / "per_attack_recall.csv", index=False)
    (results_dir / "per_attack_recall.json").write_text(
        json.dumps(per_attack_records, indent=2), encoding="utf-8"
    )

    feature_importances.to_csv(results_dir / "feature_importances.csv", index=False)

    joblib.dump(scaler, results_dir / "scaler.pkl")
    joblib.dump(feature_columns, results_dir / "feature_columns.pkl")
    joblib.dump(model, results_dir / "rf_held_out_day_model.pkl", compress=3)

    logger.info("All artifacts successfully saved to %s", results_dir)
    return experiment_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out-day generalization experiment for AI-NIDS."
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row limit per CSV for rapid debugging or validation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate file partitioning, schema, and preprocessing without training model.",
    )
    args = parser.parse_args()

    if args.validate_only:
        train_files, test_files = get_day_files(INPUT_DIR)
        print(f"Validated 8 files: {len(train_files)} train (Mon-Thu), {len(test_files)} test (Fri).")
        return

    run_experiment(max_rows_per_file=args.max_rows_per_file)


if __name__ == "__main__":
    main()
