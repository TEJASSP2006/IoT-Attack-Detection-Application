import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_data_paths(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if data_path.is_dir():
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {data_path}")
        print(f"Using {len(csv_files)} CSV files from directory: {data_path}")
        return csv_files
    raise ValueError(f"Dataset path not found: {data_path}")


def auto_detect_label_column(df: pd.DataFrame) -> str:
    preferred = [
        "label",
        "Label",
        "attack_label",
        "Attack_label",
        "attack",
        "Attack",
        "class",
        "Class",
        "target",
        "Target",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    object_like = [c for c in df.columns if df[c].dtype == "object"]
    if object_like:
        return object_like[-1]
    return df.columns[-1]


def normalize_attack_binary(y: pd.Series) -> tuple[np.ndarray, set[str]]:
    benign_tokens = {"benign", "normal", "0", "false", "no"}
    y_str = y.astype(str).str.strip()
    y_bin = (~y_str.str.lower().isin(benign_tokens)).astype(int).to_numpy()
    benign_labels = {val for val in y_str.unique() if val.lower() in benign_tokens}
    return y_bin, benign_labels


def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()
    float_cols = optimized.select_dtypes(include=["float64"]).columns
    int_cols = optimized.select_dtypes(include=["int64"]).columns
    for col in float_cols:
        optimized[col] = pd.to_numeric(optimized[col], downcast="float")
    for col in int_cols:
        optimized[col] = pd.to_numeric(optimized[col], downcast="integer")
    return optimized


def load_dataset(
    data_path: Path,
    label_column: str,
    max_rows: int,
    chunksize: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, str]:
    resolved_paths = resolve_data_paths(data_path)
    first_df = pd.read_csv(resolved_paths[0], nrows=1000)
    selected_label = label_column
    if label_column.lower() == "auto":
        selected_label = auto_detect_label_column(first_df)
        print(f"Auto-detected label column: {selected_label}")
    if selected_label not in first_df.columns:
        raise ValueError(f"Label column '{selected_label}' not found in dataset.")

    total_valid_rows = 0
    total_missing_labels = 0
    for resolved_path in resolved_paths:
        for label_chunk in pd.read_csv(resolved_path, usecols=[selected_label], chunksize=chunksize):
            valid_count = int(label_chunk[selected_label].notna().sum())
            total_valid_rows += valid_count
            total_missing_labels += int(len(label_chunk) - valid_count)

    if total_valid_rows == 0:
        raise ValueError("No rows with non-null label values were found across dataset files.")

    if max_rows > 0 and total_valid_rows > max_rows:
        sample_fraction = max_rows / total_valid_rows
        print(
            f"Sampling approximately {max_rows:,} rows from {total_valid_rows:,} usable rows "
            f"(fraction={sample_fraction:.6f}) to avoid memory exhaustion."
        )
    else:
        sample_fraction = 1.0
        print(f"Using all {total_valid_rows:,} usable rows from all CSV files.")

    rng = np.random.default_rng(random_state)
    sampled_chunks: list[pd.DataFrame] = []
    for resolved_path in resolved_paths:
        for chunk in pd.read_csv(resolved_path, chunksize=chunksize):
            if selected_label not in chunk.columns:
                raise ValueError(
                    f"Label column '{selected_label}' not found in dataset file: {resolved_path}"
                )
            chunk_valid = chunk.loc[chunk[selected_label].notna()].copy()
            if chunk_valid.empty:
                continue
            if sample_fraction < 1.0:
                keep_mask = rng.random(len(chunk_valid)) < sample_fraction
                chunk_valid = chunk_valid.loc[keep_mask]
                if chunk_valid.empty:
                    continue
            sampled_chunks.append(chunk_valid)

    if not sampled_chunks:
        raise ValueError("Sampling produced no rows. Increase --max-rows or reduce --chunksize.")

    df_all = pd.concat(sampled_chunks, ignore_index=True, sort=False)
    if max_rows > 0 and len(df_all) > max_rows:
        df_all = df_all.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    y = df_all[selected_label]
    valid_rows = y.notna()
    dropped_rows = total_missing_labels + int((~valid_rows).sum())
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing label values.")

    y = y.loc[valid_rows].reset_index(drop=True)
    X = df_all.drop(columns=[selected_label]).loc[valid_rows].reset_index(drop=True)
    X = optimize_numeric_dtypes(X)

    return X, y, selected_label


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict, float, set[str]]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_val_bin, benign_labels = normalize_attack_binary(y_val)
    val_proba = model.predict_proba(X_val)
    classes = [str(c) for c in model.classes_]
    benign_idx = [idx for idx, c in enumerate(classes) if c.lower() in {"benign", "normal", "0"}]
    if benign_idx:
        benign_prob = val_proba[:, benign_idx].max(axis=1)
    else:
        benign_prob = np.zeros(len(X_val))
    attack_score = 1.0 - benign_prob
    candidate_thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        pred_attack = (attack_score >= threshold).astype(int)
        score = f1_score(y_val_bin, pred_attack, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    metrics = {
        "accuracy": report.get("accuracy", 0.0),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        "attack_threshold": best_threshold,
        "attack_f1_at_threshold": best_f1,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "class_report": report,
    }
    return model, metrics, best_threshold, benign_labels


def save_artifacts(
    model: Pipeline,
    feature_names: list[str],
    metrics: dict,
    selected_label: str,
    attack_threshold: float,
    benign_labels: set[str],
) -> None:
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "iot_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_path": str(model_path),
        "feature_names": feature_names,
        "label_column": selected_label,
        "attack_threshold": attack_threshold,
        "benign_labels": sorted(list(benign_labels)),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CICIoT2023 attack detector model.")
    parser.add_argument(
        "--data-path",
        default="archive/wataiData/csv/CICIoT2023",
        help="Path to CICIoT2023 CSV file or directory (default: archive/wataiData/csv/CICIoT2023).",
    )
    parser.add_argument(
        "--label-column",
        default="auto",
        help="Label column name in dataset (default: auto for auto-detection).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_500_000,
        help="Maximum rows to keep in memory for training; set 0 to disable sampling (default: 1500000).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per CSV chunk while streaming files (default: 200000).",
    )
    args = parser.parse_args()

    X, y, selected_label = load_dataset(
        Path(args.data_path),
        args.label_column,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
    )
    print(f"Loaded dataset with shape: X={X.shape}, y={y.shape}, label={selected_label}")

    model, metrics, attack_threshold, benign_labels = train_model(X, y)
    save_artifacts(
        model,
        list(X.columns),
        metrics,
        selected_label,
        attack_threshold,
        benign_labels,
    )


if __name__ == "__main__":
    main()
