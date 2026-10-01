"""Prediction service for the cyber threat detection web application.

The module loads the artifacts produced by ``training/train.py`` and applies
exactly the same preprocessing to a new CSV dataset. No model is ever
(Re)trained here.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, ROOT_DIR)

from joblib import load  # noqa: E402

from training import preprocess as pp  # noqa: E402

MODELS_DIR = os.path.join(ROOT_DIR, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

PREDICTION_NORMAL = "NORMAL"
PREDICTION_MALICIOUS = "MALICIOUS"

DISPLAY_COLUMNS = (
    "Source IP",
    "Destination IP",
    "Protocol",
    "Source Port",
    "Destination Port",
    "Packet Count",
    "Byte Count",
    "Flow Duration",
    "Traffic Direction",
    "Prediction",
    "Threat Probability",
    "Attack Type",
)


class ModelNotAvailableError(Exception):
    """Raised when the trained artifacts are missing or unreadable."""


@dataclass
class PredictionResult:
    """Outcome of analysing one traffic dataset."""

    total_records: int = 0
    normal_records: int = 0
    malicious_records: int = 0
    threat_percentage: float = 0.0
    unidirectional_records: int = 0
    bidirectional_records: int = 0
    model_name: str = "unknown"
    threshold: float = 0.5
    attack_types: Dict[str, int] = field(default_factory=dict)
    protocol_distribution: Dict[str, int] = field(default_factory=dict)
    direction_distribution: Dict[str, int] = field(default_factory=dict)
    probability_histogram: Dict[str, int] = field(default_factory=dict)
    top_sources: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth_accuracy: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    warnings: List[str] = field(default_factory=list)
    records_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    display_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "normal": self.normal_records,
            "malicious": self.malicious_records,
            "threat_percentage": round(self.threat_percentage, 2),
            "unidirectional_records": self.unidirectional_records,
            "bidirectional_records": self.bidirectional_records,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "attack_types": self.attack_types,
            "protocol_distribution": self.protocol_distribution,
            "direction_distribution": self.direction_distribution,
            "probability_histogram": self.probability_histogram,
            "top_sources": self.top_sources,
            "ground_truth_accuracy": self.ground_truth_accuracy,
            "warnings": self.warnings,
        }


class ThreatPredictor:
    """Loads the trained artifacts and scores traffic records."""

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.metadata: Optional[Dict[str, Any]] = None
        self.model = None
        self.attack_bundle: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        metadata_path = os.path.join(self.models_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            self.error = (
                "No trained model was found. Run 'python training/train.py' to build the "
                "models before using the prediction service."
            )
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
            best_file = self.metadata.get("best_model_file") or "best_model.pkl"
            model_path = os.path.join(self.models_dir, best_file)
            if not os.path.exists(model_path):
                model_path = os.path.join(self.models_dir, "best_model.pkl")
            self.model = load(model_path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
            self.error = f"The trained model could not be loaded: {exc}"
            return

        bundle_path = os.path.join(self.models_dir, "attack_classifier_bundle.pkl")
        if os.path.exists(bundle_path):
            try:
                self.attack_bundle = load(bundle_path)
            except Exception:  # noqa: BLE001 - optional component
                self.attack_bundle = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.metadata is not None

    def require_model(self) -> None:
        if not self.available:
            raise ModelNotAvailableError(self.error or "Trained model unavailable.")

    def predict_frame(self, frame: pd.DataFrame) -> PredictionResult:
        """Score a raw traffic DataFrame that has already been canonicalised."""
        self.require_model()
        metadata = self.metadata or {}

        feature_columns: List[str] = list(metadata.get("feature_columns", []))
        engineered = set(metadata.get("engineered_features", []))
        available = [column for column in feature_columns if column in frame.columns]
        raw_available = [column for column in available if column not in engineered]
        missing = [column for column in feature_columns if column not in frame.columns]
        if not raw_available:
            raise pp.DatasetError(
                "The uploaded dataset shares no traffic feature with the trained model, so predictions "
                "would be meaningless. Expected columns include: "
                + ", ".join([column for column in feature_columns if column not in engineered][:10])
                + ("..." if len(feature_columns) > 10 else "")
                + ". Re-train with 'python training/train.py' on a compatible dataset."
            )
        if not available:
            raise pp.DatasetError(
                "None of the features required by the trained model were found in the uploaded dataset."
            )
        if missing:
            for column in missing:
                frame[column] = np.nan

        X = frame[feature_columns]
        predictions = np.asarray(self.model.predict(X))
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            try:
                probabilities = np.asarray(self.model.predict_proba(X))
            except Exception:  # noqa: BLE001 - optional
                probabilities = None

        threat_probability = self._threat_probability(predictions, probabilities)
        threshold = float(metadata.get("threshold", 0.5))
        predicted_binary = (threat_probability >= threshold).astype(int) if probabilities is not None else predictions
        labels = np.where(predicted_binary == 1, PREDICTION_MALICIOUS, PREDICTION_NORMAL)

        attack_types = np.full(len(frame), "n/a", dtype=object)
        attack_warnings: List[str] = []
        if self.attack_bundle is not None:
            try:
                attack_pipeline = self.attack_bundle["pipeline"]
                classes = list(self.attack_bundle.get("classes", []))
                attack_features = list(self.attack_bundle.get("feature_columns", feature_columns))
                attack_frame = frame[[c for c in attack_features if c in frame.columns]]
                raw = np.asarray(attack_pipeline.predict(attack_frame))
                attack_types = np.array([str(classes[int(index)]) for index in raw], dtype=object)
            except Exception as exc:  # noqa: BLE001 - optional component
                attack_warnings.append(f"Attack typing unavailable: {exc}")

        result = PredictionResult()
        result.model_name = str(metadata.get("best_model", "unknown model"))
        result.threshold = threshold
        result.warnings.extend(list(metadata.get("warnings", [])))
        result.warnings.extend(attack_warnings)
        if missing:
            result.warnings.append(
                "Feature(s) absent from the uploaded dataset were imputed with training values: "
                + ", ".join(missing[:10])
                + ("..." if len(missing) > 10 else "")
            )
        result.total_records = int(len(frame))
        result.malicious_records = int((labels == PREDICTION_MALICIOUS).sum())
        result.normal_records = int((labels == PREDICTION_NORMAL).sum())
        result.threat_percentage = (
            100.0 * result.malicious_records / result.total_records if result.total_records else 0.0
        )

        direction = frame["traffic_direction"] if "traffic_direction" in frame else pd.Series(["UNKNOWN"] * len(frame))
        result.unidirectional_records = int((direction == pp.DIRECTION_UNIDIRECTIONAL).sum())
        result.bidirectional_records = int(direction.isin([pp.DIRECTION_FORWARD, pp.DIRECTION_REVERSE]).sum())
        result.direction_distribution = {
            str(key): int(value) for key, value in direction.value_counts().items()
        }

        if "protocol" in frame:
            result.protocol_distribution = {
                str(key): int(value) for key, value in frame["protocol"].astype(str).value_counts().head(10).items()
            }

        detected = pd.Series(labels, index=frame.index)
        result.attack_types = {
            str(key): int(value)
            for key, value in pd.Series(attack_types)[detected == PREDICTION_MALICIOUS].value_counts().items()
        }

        bins = np.linspace(0.0, 1.0, 11)
        histogram, edges = np.histogram(threat_probability, bins=bins)
        result.probability_histogram = {
            f"{edges[index]:.1f}-{edges[index + 1]:.1f}": int(count)
            for index, count in enumerate(histogram)
            if count > 0
        }

        source_ip = frame["source_ip"].astype(str) if "source_ip" in frame else pd.Series(["unknown"] * len(frame))
        top = source_ip.value_counts().head(6)
        result.top_sources = [{"source": str(name), "records": int(count)} for name, count in top.items()]

        records = pd.DataFrame(
            {
                "Source IP": source_ip.values,
                "Destination IP": frame["destination_ip"].astype(str).values
                if "destination_ip" in frame
                else ["unknown"] * len(frame),
                "Protocol": frame["protocol"].astype(str).values
                if "protocol" in frame
                else ["unknown"] * len(frame),
                "Source Port": frame["source_port"].values if "source_port" in frame else ["-"] * len(frame),
                "Destination Port": frame["destination_port"].values
                if "destination_port" in frame
                else ["-"] * len(frame),
                "Packet Count": frame["packet_count"].values if "packet_count" in frame else [0] * len(frame),
                "Byte Count": frame["byte_count"].values if "byte_count" in frame else [0] * len(frame),
                "Flow Duration": frame["flow_duration"].values if "flow_duration" in frame else [0] * len(frame),
                "Traffic Direction": direction.values,
                "Prediction": labels,
                "Threat Probability": np.round(threat_probability * 100.0, 2),
                "Attack Type": attack_types,
            }
        )
        result.records_frame = records
        result.display_table = records.head(250).copy()

        self._score_against_ground_truth(frame, labels, result)
        return result

    @staticmethod
    def _threat_probability(predictions: np.ndarray, probabilities: Optional[np.ndarray]) -> np.ndarray:
        if probabilities is not None and probabilities.ndim == 2:
            if probabilities.shape[1] == 2:
                return probabilities[:, 1]
            return probabilities.max(axis=1)
        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _score_against_ground_truth(frame: pd.DataFrame, labels: np.ndarray, result: PredictionResult) -> None:
        if "Label_normalised" not in frame.columns:
            return
        truth = frame["Label_normalised"].astype(str)
        expected = np.where(pp.binary_target(truth) == 1, PREDICTION_MALICIOUS, PREDICTION_NORMAL)
        correct = int((expected == labels).sum())
        result.ground_truth_accuracy = round(100.0 * correct / len(labels), 2) if len(labels) else None

        matrix = pd.crosstab(pd.Series(expected, name="actual"), pd.Series(labels, name="predicted"))
        matrix = matrix.reindex(index=[PREDICTION_NORMAL, PREDICTION_MALICIOUS],
                                columns=[PREDICTION_NORMAL, PREDICTION_MALICIOUS], fill_value=0)
        result.confusion_matrix = matrix.to_numpy().tolist()

    def analyse_csv(self, path: str) -> PredictionResult:
        """Load a CSV from disk, apply the training preprocessing and score it."""
        raw = pp.read_dataset(path)
        canonical, _, unmapped = pp.canonicalize_columns(raw)
        label_column = pp.find_label_column(canonical.columns)
        frame = pp.clean_frame(canonical, label_column)
        frame = frame.drop_duplicates().reset_index(drop=True)
        if label_column is not None:
            frame["Label_normalised"] = frame[label_column].map(pp.normalise_label)
            frame = frame.drop(columns=[label_column])
        else:
            frame["Label_normalised"] = "unlabelled"
        frame = pp.add_engineered_features(frame)
        result = self.predict_frame(frame)
        if unmapped:
            result.warnings.append(
                "Columns ignored during prediction: " + ", ".join(unmapped[:10])
                + ("..." if len(unmapped) > 10 else "")
            )
        return result


_predictor: Optional[ThreatPredictor] = None


def get_predictor() -> ThreatPredictor:
    """Return the process-wide predictor instance (models load only once)."""
    global _predictor
    if _predictor is None:
        _predictor = ThreatPredictor()
    return _predictor


def reload_predictor() -> ThreatPredictor:
    """Force a reload, for example after re-running the training pipeline."""
    global _predictor
    _predictor = ThreatPredictor()
    return _predictor