"""End-to-end training pipeline for the cyber threat detection system.

Run with::

    python training/train.py --dataset data/traffic_dataset.csv

The script validates and cleans the dataset, engineers the unidirectional
traffic features, fits the preprocessing pipeline on the training split only,
trains and compares several classifiers, evaluates every model with the full
metric bundle (accuracy, precision, recall, F1, ROC-AUC, confusion matrix),
saves each pipeline with joblib and writes ``models/metadata.json`` plus
Matplotlib/Seaborn figures consumed by the web dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", message=".*parallel_backend.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from joblib import dump  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from training import evaluate as ev  # noqa: E402
from training import preprocess as pp  # noqa: E402

try:  # XGBoost is optional.
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    XGBClassifier = None  # type: ignore
    XGBOOST_AVAILABLE = False

try:  # SHAP is optional explainability support.
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    shap = None  # type: ignore
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
TEST_SIZE = 0.2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(ROOT_DIR, "data", "traffic_dataset.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "static", "images", "generated")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def build_models(random_state: int, positive_weight: float) -> Dict[str, Any]:
    """Instantiate every classifier that will take part in the comparison."""
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state, n_jobs=None
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    return models


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Imputation + scaling / one-hot encoding fitted on the training split only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:  # pragma: no cover - defensive
        names = []
    return [str(name) for name in names]


def feature_importance_table(estimator: Any, names: List[str], top: int = 15) -> List[Dict[str, Any]]:
    """Feature importance using the method native to each estimator family."""
    table: List[Dict[str, Any]] = []
    importance: Optional[np.ndarray] = None

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        importance = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)

    if importance is None or importance.size == 0:
        return table

    if names and len(names) == importance.size:
        labels = names
    else:
        labels = [f"feature_{i}" for i in range(importance.size)]

    total = float(importance.sum()) or 1.0
    order = np.argsort(importance)[::-1][:top]
    for rank, index in enumerate(order, start=1):
        value = float(importance[index])
        table.append(
            {
                "rank": rank,
                "feature": labels[index],
                "importance": round(value, 6),
                "importance_pct": round(100.0 * value / total, 2),
            }
        )
    return table


def shap_summary(estimator: Any, preprocessor: ColumnTransformer, X_sample: pd.DataFrame,
                names: List[str], top: int = 10) -> Optional[Dict[str, Any]]:
    """Optional SHAP explanation; the pipeline works without SHAP installed."""
    if not SHAP_AVAILABLE:
        return None
    try:
        transformed = preprocessor.transform(X_sample)
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        mean_abs = np.abs(np.asarray(values, float)).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top]
        labels = names if names and len(names) == mean_abs.size else [f"feature_{i}" for i in range(mean_abs.size)]
        return {
            "available": True,
            "mean_abs_shap": [
                {"feature": labels[i], "value": round(float(mean_abs[i]), 6)} for i in order
            ],
        }
    except Exception as exc:  # pragma: no cover - SHAP is best effort
        return {"available": False, "error": str(exc)}


def load_and_prepare(dataset_path: str) -> Tuple[pd.DataFrame, pp.DatasetReport, str, bool]:
    """Read, canonicalise, clean, de-duplicate and engineer the dataset."""
    raw = pp.read_dataset(dataset_path)
    canonical, mapping, unmapped = pp.canonicalize_columns(raw)

    label_column = pp.find_label_column(canonical.columns)
    if label_column is None:
        raise pp.DatasetError(
            "No target/label column was detected. Expected one of: "
            + ", ".join(pp.LABEL_ALIASES[:8])
            + ". Add a column such as 'Label' with values Normal / <attack>."
        )

    report = pp.profile_dataset(canonical, label_column)
    report.mapped_columns = mapping
    unmapped = [column for column in unmapped if column != label_column]
    report.unmapped_columns = unmapped
    if unmapped:
        report.warnings.append(
            "Columns not used for training: " + ", ".join(unmapped[:12])
            + ("..." if len(unmapped) > 12 else "")
        )

    frame = pp.clean_frame(canonical, label_column)
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        frame = frame.drop_duplicates().reset_index(drop=True)
    report.duplicate_rows = duplicates

    frame["Label_normalised"] = frame[label_column].map(pp.normalise_label)
    frame = frame.drop(columns=[label_column])
    frame = frame[frame["Label_normalised"] != "unknown"].reset_index(drop=True)

    frame = pp.add_engineered_features(frame)
    direction = frame["traffic_direction"]
    report.unidirectional_records = int((direction == pp.DIRECTION_UNIDIRECTIONAL).sum())
    report.bidirectional_records = int(direction.isin([pp.DIRECTION_FORWARD, pp.DIRECTION_REVERSE]).sum())
    report.classes = sorted(frame["Label_normalised"].unique().tolist())
    report.rows = int(frame.shape[0])

    has_attacks = bool((frame["binary_target"] if "binary_target" in frame else
                        pp.binary_target(frame["Label_normalised"])).sum() > 0)
    return frame, report, label_column, has_attacks


def train() -> Dict[str, Any]:
    """Run the full training pipeline and persist every artefact."""
    parser = argparse.ArgumentParser(description="Train the cyber threat detection models.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to the CSV traffic dataset.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Fraction held out for testing.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Reproducibility seed.")
    args = parser.parse_args()

    for directory in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        os.makedirs(directory, exist_ok=True)

    started = time.time()
    print("[1/8] Loading dataset:", args.dataset)
    frame, report, label_column, has_attacks = load_and_prepare(args.dataset)
    print(f"      rows={report.rows} columns={report.columns} label='{label_column}'")
    print(f"      classes: {report.classes}")
    print(f"      unidirectional flows: {report.unidirectional_records}")
    for warning in report.warnings:
        print("      warning:", warning)

    print("[2/8] Separating features and target")
    feature_columns, numeric_features, categorical_features = pp.determine_feature_columns(
        frame, "Label_normalised"
    )
    feature_columns = [column for column in feature_columns if column != "Label_normalised"]
    numeric_features = [column for column in numeric_features if column != "Label_normalised"]
    categorical_features = [column for column in categorical_features if column != "Label_normalised"]
    print(f"      using {len(feature_columns)} features "
          f"({len(numeric_features)} numeric / {len(categorical_features)} categorical)")

    X = frame[feature_columns]
    y_binary = pp.binary_target(frame["Label_normalised"])
    encoder = LabelEncoder()
    y_multiclass = encoder.fit_transform(frame["Label_normalised"].tolist())
    class_names = [str(name) for name in encoder.classes_]
    binary_class_names = ["Normal", "Attack"]

    print("[3/8] Splitting train/test (stratified, random_state=%d)" % args.random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=args.test_size, random_state=args.random_state, stratify=y_binary
    )
    print(f"      train={len(X_train)} test={len(X_test)}")
    print(f"      train class balance: normal={int((y_train == 0).sum())} attack={int((y_train == 1).sum())}")

    print("[4/8] Fitting preprocessing on the training split only")
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = transformed_feature_names(preprocessor)
    print(f"      transformed train matrix: {X_train_t.shape[0]} x {X_train_t.shape[1]}")
    print(f"      transformed test matrix:  {X_test_t.shape[0]} x {X_test_t.shape[1]}")

    positive_weight = float((y_train == 0).sum()) / float(max((y_train == 1).sum(), 1))
    models = build_models(args.random_state, positive_weight)

    print("[5/8] Training and evaluating models")
    results: List[Dict[str, Any]] = []
    confusion_images: Dict[str, str] = {}
    roc_images: Dict[str, str] = {}
    importance_tables: Dict[str, List[Dict[str, Any]]] = {}
    saved_models: Dict[str, str] = {}
    best_pipeline: Optional[Pipeline] = None
    best_name = ""
    best_score = -np.inf

    for name, estimator in models.items():
        model_started = time.time()
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - model_started

        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception:  # pragma: no cover - defensive
                y_proba = None

        metrics = ev.classification_metrics(
            y_test, y_pred, y_proba, binary=True, class_names=binary_class_names
        )
        metrics["model_name"] = name
        metrics["train_seconds"] = round(elapsed, 3)
        metrics["xgboost"] = "XGBClassifier"

        estimator_fitted = pipeline.named_steps["classifier"]
        importance_tables[name] = feature_importance_table(estimator_fitted, feature_names)

        confusion_images[name] = ev.plot_confusion_matrix(
            metrics["confusion_matrix"], metrics["labels"], name, FIGURES_DIR
        )
        if y_proba is not None and y_proba.shape[1] == 2:
            roc_images[name] = ev.plot_roc(metrics["roc_curve"], metrics["roc_auc"], name, FIGURES_DIR)

        dump(pipeline, os.path.join(MODELS_DIR, f"{slugify(name)}.pkl"))
        saved_models[name] = f"{slugify(name)}.pkl"

        score = metrics["f1_score"]
        if metrics["roc_auc"] and not np.isnan(metrics["roc_auc"]):
            score = 0.5 * score + 0.5 * metrics["roc_auc"]
        if score > best_score:
            best_score, best_name, best_pipeline = score, name, pipeline

        results.append(ev.sanitise_metrics(metrics))
        print(f"      {name:<20} acc={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f} "
              f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is None else round(metrics['roc_auc'], 4)} "
              f"({elapsed:.1f}s)")

    if best_pipeline is None:  # pragma: no cover - defensive
        raise RuntimeError("No model could be trained.")

    dump(preprocessor, os.path.join(MODELS_DIR, "scaler.pkl"))
    dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.pkl"))

    print("[6/8] Training the attack-type (multiclass) classifier")
    attack_classifier: Dict[str, Any] = {"available": False}
    attack_distribution: Dict[str, int] = {}
    for label in class_names:
        attack_distribution[label] = int((frame["Label_normalised"] == label).sum())

    if len(class_names) > 2:
        Xa_train, Xa_test, ya_train, ya_test = train_test_split(
            X, y_multiclass, test_size=args.test_size, random_state=args.random_state, stratify=y_multiclass
        )
        if XGBOOST_AVAILABLE:
            attack_estimator = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist",
                n_jobs=-1, random_state=args.random_state,
            )
        else:
            attack_estimator = GradientBoostingClassifier(random_state=args.random_state)
        attack_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", attack_estimator)])
        attack_pipeline.fit(Xa_train, ya_train)
        ya_pred = attack_pipeline.predict(Xa_test)
        ya_proba = attack_pipeline.predict_proba(Xa_test)
        attack_metrics = ev.classification_metrics(
            ya_test, ya_pred, ya_proba, binary=False, class_names=class_names
        )
        dump(attack_pipeline, os.path.join(MODELS_DIR, "attack_classifier.pkl"))
        dump(
            {
                "classes": class_names,
                "pipeline": attack_pipeline,
                "feature_columns": feature_columns,
            },
            os.path.join(MODELS_DIR, "attack_classifier_bundle.pkl"),
        )
        attack_classifier = {
            "available": True,
            "classes": class_names,
            "metrics": ev.sanitise_metrics(attack_metrics),
            "model_file": "attack_classifier.pkl",
            "confusion_matrix_image": ev.plot_confusion_matrix(
                attack_metrics["confusion_matrix"], class_names, "Attack Type Classifier", FIGURES_DIR
            ),
        }
        print(f"      attack-type accuracy={attack_metrics['accuracy']:.4f} f1={attack_metrics['f1_score']:.4f}")
    else:
        dump(
            {"classes": class_names, "pipeline": best_pipeline, "feature_columns": feature_columns},
            os.path.join(MODELS_DIR, "attack_classifier_bundle.pkl"),
        )
        attack_classifier = {
            "available": True,
            "reused_binary_model": True,
            "classes": class_names,
            "model_file": f"{slugify(best_name)}.pkl",
            "metrics": None,
            "confusion_matrix_image": None,
        }
        print("      dataset is binary (Normal vs Attack): reusing the binary model for attack typing")

    print("[7/8] Explainability (feature importance + optional SHAP)")
    best_estimator = best_pipeline.named_steps["classifier"]
    shap_summary_data = shap_summary(best_estimator, preprocessor, X_test.head(200), feature_names)

    comparison_image = ev.plot_model_comparison(
        [{"model_name": r["model_name"], **{k: (r[k] or 0.0) for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")}}
         for r in results],
        FIGURES_DIR,
    )

    metadata: Dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_path": os.path.relpath(args.dataset, ROOT_DIR).replace("\\", "/"),
        "dataset_name": os.path.basename(args.dataset),
        "label_column": label_column,
        "class_names": class_names,
        "binary_class_names": binary_class_names,
        "attack_distribution": attack_distribution,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "engineered_features": list(pp.ENGINEERED_NUMERIC) + list(pp.ENGINEERED_CATEGORICAL),
        "identifier_columns": list(pp.IDENTIFIER_COLUMNS),
        "mapped_columns": report.mapped_columns,
        "unmapped_columns": report.unmapped_columns,
        "duplicate_rows_removed": report.duplicate_rows,
        "missing_values_found": report.missing_values,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "normal_records": int((frame["Label_normalised"].map(pp.is_normal_class)).sum()),
        "unidirectional_records": int((frame["traffic_direction"] == pp.DIRECTION_UNIDIRECTIONAL).sum()),
        "bidirectional_records": int(
            frame["traffic_direction"].isin([pp.DIRECTION_FORWARD, pp.DIRECTION_REVERSE]).sum()
        ),
        "models": results,
        "model_files": saved_models,
        "best_model": best_name,
        "best_model_file": f"{slugify(best_name)}.pkl",
        "scaler_file": "scaler.pkl",
        "confusion_images": confusion_images,
        "roc_images": roc_images,
        "comparison_image": comparison_image,
        "feature_importance": importance_tables.get(best_name, []),
        "feature_importance_method": (
            "impurity based (feature_importances_)"
            if hasattr(best_estimator, "feature_importances_")
            else "coefficient magnitude (|coef_|)"
        ),
        "shap": shap_summary_data,
        "attack_classifier": attack_classifier,
        "xgboost_available": XGBOOST_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
        "imbalance_strategy": (
            "class_weight='balanced' (Logistic Regression, Decision Tree, Random Forest), "
            "scale_pos_weight for XGBoost and a stratified split so the rare attack class "
            "stays represented in the test set."
        ),
        "threshold": 0.5,
        "training_seconds": round(time.time() - started, 2),
    }

    print("[8/8] Writing models/metadata.json")
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2)

    with open(os.path.join(REPORTS_DIR, "training_report.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Artifacts written to {os.path.relpath(MODELS_DIR, ROOT_DIR)} and "
          f"{os.path.relpath(FIGURES_DIR, ROOT_DIR)}")
    return metadata


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy types and NaN values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if np.isnan(numeric) else numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    train()