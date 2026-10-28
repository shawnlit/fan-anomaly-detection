#!/usr/bin/env python3
"""
INA219-only fan condition pipeline.

Inputs (in the same directory by default):
  fan_raw_dataset(1).csv                 -> NORMAL
  fan_raw_dataset_airflow_restricted.csv -> AIRFLOW_RESTRICTED

Outputs:
  ina_windowed_dataset.csv
  ina_decision_tree.joblib
  ina_model_report.txt

This script uses voltage, current, and calculated power only.
It deliberately does NOT use ina_power_mW because it was inconsistent
with voltage × current in the examined recordings.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

BASE_DIR = Path(__file__).resolve().parent
NORMAL_FILE = BASE_DIR / "fan_raw_dataset_normal.csv"
RESTRICTED_FILE = BASE_DIR / "fan_raw_dataset_airflow_restricted.csv"

WINDOW_SECONDS = 5
ELECTRICAL_SAMPLE_HZ = 10  # INA219 updates around every 100 ms
TEST_RUN_NUMBER = 5         # Hold out run 5 from each class for testing

RAW_REQUIRED = [
    "run_id", "label", "timestamp_ms",
    "voltage_V", "current_mA", "power_calc_mW"
]

SIGNALS = ["voltage_V", "current_mA", "power_calc_mW"]


def load_one_file(path: Path, expected_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Put it beside this script or edit the file path."
        )

    df = pd.read_csv(path)
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[RAW_REQUIRED].copy()
    df["label"] = expected_label
    df["source_file"] = path.name

    for col in ["timestamp_ms", *SIGNALS, "run_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", *SIGNALS, "run_id"])
    df["run_id"] = df["run_id"].astype(int)

    # Create globally unique run IDs so NORMAL run 1 and RESTRICTED run 1
    # cannot accidentally be treated as the same recording.
    df["group_id"] = expected_label + "_run_" + df["run_id"].astype(str)

    return df


def make_window_features(raw: pd.DataFrame) -> pd.DataFrame:
    all_windows = []
    samples_per_window = WINDOW_SECONDS * ELECTRICAL_SAMPLE_HZ

    for group_id, run in raw.groupby("group_id", sort=True):
        run = run.sort_values("timestamp_ms").copy()
        run["elapsed_s"] = (run["timestamp_ms"] - run["timestamp_ms"].iloc[0]) / 1000.0

        # The logger prints IMU rows at ~200 Hz, but INA219 values update
        # much more slowly. Keep one electrical observation per 100 ms bin.
        run["sample_bin"] = np.floor(
            run["elapsed_s"] * ELECTRICAL_SAMPLE_HZ
        ).astype(int)
        electrical = (
            run.groupby("sample_bin", as_index=False)
            .agg({
                "elapsed_s": "last",
                "voltage_V": "last",
                "current_mA": "last",
                "power_calc_mW": "last",
                "label": "first",
                "source_file": "first",
                "run_id": "first",
                "group_id": "first",
            })
            .sort_values("sample_bin")
            .reset_index(drop=True)
        )

        # Non-overlapping 5-second windows. Discard an incomplete final window.
        electrical["window_id"] = (
            electrical["sample_bin"] // samples_per_window
        ).astype(int)

        for window_id, window in electrical.groupby("window_id", sort=True):
            if len(window) < samples_per_window:
                continue

            features = {
                "group_id": group_id,
                "run_id": int(window["run_id"].iloc[0]),
                "source_file": window["source_file"].iloc[0],
                "label": window["label"].iloc[0],
                "window_id": int(window_id),
                "window_start_s": float(window["elapsed_s"].min()),
                "window_end_s": float(window["elapsed_s"].max()),
                "samples_in_window": int(len(window)),
            }
            for signal in SIGNALS:
                values = window[signal].astype(float)
                features[f"{signal}_mean"] = float(values.mean())
                features[f"{signal}_std"] = float(values.std(ddof=0))
                features[f"{signal}_min"] = float(values.min())
                features[f"{signal}_max"] = float(values.max())

            all_windows.append(features)

    if not all_windows:
        raise ValueError("No complete windows created. Check timestamps and input data.")

    return pd.DataFrame(all_windows)


def main():
    normal = load_one_file(NORMAL_FILE, "NORMAL")
    restricted = load_one_file(RESTRICTED_FILE, "AIRFLOW_RESTRICTED")
    raw = pd.concat([normal, restricted], ignore_index=True)

    # Basic data-quality checks
    print("Raw rows:")
    print(raw.groupby(["label", "group_id"]).size().to_string())
    print()

    windows = make_window_features(raw)
    windows_path = BASE_DIR / "ina_windowed_dataset.csv"
    windows.to_csv(windows_path, index=False)

    feature_cols = [
        f"{signal}_{stat}"
        for signal in SIGNALS
        for stat in ["mean", "std", "min", "max"]
    ]

    # Hold out run 5 from EACH class. This prevents adjacent windows from the
    # same physical recording appearing in both train and test sets.
    test_mask = windows["run_id"].eq(TEST_RUN_NUMBER)
    train = windows.loc[~test_mask].copy()
    test = windows.loc[test_mask].copy()

    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise ValueError(
            "Train/test split does not contain both classes. Ensure both files "
            "have runs 1–5, or change TEST_RUN_NUMBER."
        )

    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]

    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(
        y_test, predictions, labels=["NORMAL", "AIRFLOW_RESTRICTED"]
    )

    model_path = BASE_DIR / "ina_decision_tree.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "window_seconds": WINDOW_SECONDS,
            "electrical_sample_hz": ELECTRICAL_SAMPLE_HZ,
            "classes": list(model.classes_),
            "note": "INA219-only prototype; power feature is voltage × current.",
        },
        model_path,
    )

    report_path = BASE_DIR / "ina_model_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("INA219-ONLY FAN CONDITION MODEL\n")
        f.write("===============================\n\n")
        f.write(f"Window length: {WINDOW_SECONDS} seconds\n")
        f.write("Electrical sampling used: approximately 10 Hz\n")
        f.write("Signals: voltage_V, current_mA, power_calc_mW\n")
        f.write("ina_power_mW intentionally excluded due to inconsistent readings.\n")
        f.write("Test split: run 5 from each class; other runs used for training.\n\n")
        f.write(f"Window counts by label:\n{windows['label'].value_counts().to_string()}\n\n")
        f.write(f"Train windows: {len(train)}\nTest windows: {len(test)}\n")
        f.write(f"Test accuracy: {accuracy:.3f}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix (rows=true, columns=predicted; NORMAL, AIRFLOW_RESTRICTED):\n")
        f.write(str(matrix) + "\n\n")
        f.write("Decision tree rules:\n")
        f.write(export_text(model, feature_names=feature_cols))

    print(f"Created {windows_path.name} ({len(windows)} windows)")
    print(f"Created {model_path.name}")
    print(f"Created {report_path.name}")
    print()
    print("Window counts by label:")
    print(windows["label"].value_counts().to_string())
    print(f"\nTrain windows: {len(train)} | Test windows: {len(test)}")
    print(f"Test accuracy (held-out runs): {accuracy:.3f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix [NORMAL, AIRFLOW_RESTRICTED]:")
    print(matrix)
    print("\nImportant: this is an initial experiment, not proof of generalization.")
    print("Repeat with different held-out runs / more physical runs before deployment.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
