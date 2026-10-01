"""Dataset loading, canonical column mapping, cleaning and feature engineering.

The module is shared by the training pipeline (``training/train.py``) and the
prediction pipeline (``utils/prediction.py``) so that a dataset uploaded through
the web application is always transformed exactly like the training data.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NORMAL_LABEL_TOKENS = (
    "normal",
    "benign",
    "legitimate",
    "clean",
    "non-attack",
    "nonattack",
    "non_attack",
    "no attack",
    "none",
    "0",
)

ATTACK_KEYWORD_TOKENS = (
    "dos",
    "ddos",
    "scan",
    "brute",
    "botnet",
    "bot",
    "infiltration",
    "malware",
    "attack",
    "flood",
    "probe",
    "spoof",
    "exploit",
    "web",
    "ftp",
    "ssh",
    "sql",
)

DIRECTION_FORWARD = "FORWARD"
DIRECTION_REVERSE = "REVERSE"
DIRECTION_UNIDIRECTIONAL = "UNIDIRECTIONAL"
DIRECTION_UNKNOWN = "UNKNOWN"
DIRECTION_VALUES = (
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
    DIRECTION_UNIDIRECTIONAL,
    DIRECTION_UNKNOWN,
)

CANONICAL_FEATURES: Tuple[str, ...] = (
    "flow_id",
    "source_ip",
    "destination_ip",
    "source_port",
    "destination_port",
    "protocol",
    "protocol_type",
    "packet_count",
    "byte_count",
    "packet_length",
    "flow_duration",
    "forward_packets",
    "forward_bytes",
    "reverse_packets",
    "reverse_bytes",
    "tcp_flags",
    "inter_arrival_time",
    "traffic_direction",
)

ALIASES: Dict[str, Tuple[str, ...]] = {
    "flow_id": ("flow id", "flowid", "flow", "flow identifier", "sid", "session id", "id"),
    "source_ip": (
        "source ip",
        "source ip address",
        "srcip",
        "src ip",
        "src ip address",
        "src addr",
        "source address",
        "sender ip",
        "sourceaddr",
        "ip src",
    ),
    "destination_ip": (
        "destination ip",
        "destination ip address",
        "dstip",
        "dst ip",
        "dest ip",
        "dst ip address",
        "dst addr",
        "destination address",
        "receiver ip",
        "destinationaddr",
        "ip dst",
    ),
    "source_port": (
        "source port",
        "src port",
        "srcport",
        "sport",
        "sourceport",
        "source port number",
        "sportnumber",
        "src port number",
    ),
    "destination_port": (
        "destination port",
        "dst port",
        "dest port",
        "dstport",
        "dport",
        "destinationport",
        "destination port number",
        "dstportnumber",
    ),
    "protocol": (
        "protocol",
        "proto",
        "ip protocol",
        "transport protocol",
        "protocol name",
        "protocol type name",
        "application",
        "service",
    ),
    "protocol_type": (
        "protocol type",
        "proto type",
        "protocoltype",
        "protocol number",
        "protocol id",
        "ip proto",
        "protocol numeric",
    ),
    "packet_count": (
        "packet count",
        "packets",
        "packetcount",
        "num packets",
        "number of packets",
        "packet total",
        "total packets",
        "pkt count",
        "no of packets",
        "all_packets",
    ),
    "byte_count": (
        "byte count",
        "bytes",
        "bytecount",
        "num bytes",
        "number of bytes",
        "total bytes",
        "byte total",
        "bytes count",
        "no of bytes",
        "total length",
        "flow bytes",
        "total byte",
    ),
    "packet_length": (
        "packet length",
        "packetlength",
        "packet size",
        "length",
        "pkt length",
        "pkt len",
        "frame length",
        "avg packet length",
        "average packet length",
        "avg packet size",
        "average packet size",
        "mean packet length",
    ),
    "flow_duration": (
        "flow duration",
        "duration",
        "flowduration",
        "flow duration us",
        "flow duration microseconds",
        "duration flow",
        "flow time",
        "elapsed time",
        "timestamp",
    ),
    "forward_packets": (
        "forward packets",
        "fwd packets",
        "packets forward",
        "forwardpacketcount",
        "fwdpackets",
        "fwd packet count",
        "packets fwd",
        "src to dst packets",
        "packets forward direction",
    ),
    "forward_bytes": (
        "forward bytes",
        "fwd bytes",
        "bytes forward",
        "forwardbytecount",
        "fwdbytes",
        "fwd byte count",
        "bytes fwd",
        "src to dst bytes",
    ),
    "reverse_packets": (
        "reverse packets",
        "backward packets",
        "rev packets",
        "packets reverse",
        "reversepacketcount",
        "rpackets",
        "revpackets",
        "packets bwd",
        "dst to src packets",
        "backwardpacketcount",
    ),
    "reverse_bytes": (
        "reverse bytes",
        "backward bytes",
        "rev bytes",
        "bytes reverse",
        "reversebytecount",
        "rbytes",
        "revbytes",
        "bytes bwd",
        "dst to src bytes",
    ),
    "tcp_flags": (
        "tcp flags",
        "flags",
        "tcpflag",
        "tcp flags count",
        "tcp flag value",
        "flags count",
        "flag count",
    ),
    "inter_arrival_time": (
        "inter arrival time",
        "interarrivaltime",
        "inter packet arrival time",
        "iat",
        "mean iat",
        "flow iat mean",
        "packet iat",
        "inter arrival",
    ),
    "traffic_direction": (
        "traffic direction",
        "direction",
        "flow direction",
        "flowdirection",
        "dir",
        "direction label",
        "flow dir",
        "traffic flow direction",
    ),
}

LABEL_ALIASES: Tuple[str, ...] = (
    "label",
    "class",
    "attack",
    "attack type",
    "attacktype",
    "attack label",
    "category",
    "classname",
    "class name",
    "class label",
    "target",
    "y",
    "activity",
    "traffic class",
    "ground truth",
    "groundtruth",
    "type",
    "label class",
)

IDENTIFIER_COLUMNS: Tuple[str, ...] = ("flow_id", "source_ip", "destination_ip")

ENGINEERED_NUMERIC: Tuple[str, ...] = (
    "is_unidirectional",
    "is_bidirectional",
    "bytes_per_packet",
    "forward_packets_ratio",
    "reverse_packets_ratio",
    "forward_bytes_ratio",
    "packets_per_second",
    "bytes_per_second",
    "flow_duration_seconds",
    "inter_arrival_per_packet",
    "tcp_flag_count",
    "destination_port_privileged",
    "unique_port_ratio",
    "port_span",
)

ENGINEERED_CATEGORICAL: Tuple[str, ...] = ("traffic_direction", "source_subnet")

LOOKUP: Dict[str, str] = {}
for _canonical, _aliases in ALIASES.items():
    LOOKUP[_canonical] = _canonical
    for _alias in _aliases:
        LOOKUP[_alias] = _canonical

LABEL_LOOKUP: Dict[str, str] = {"label column": "label column"}
for _alias in LABEL_ALIASES:
    LABEL_LOOKUP[_alias] = "label column"


class DatasetError(Exception):
    """Raised when a dataset cannot be used for training or prediction."""


@dataclass
class DatasetReport:
    """Structured description of a loaded dataset."""

    rows: int = 0
    columns: int = 0
    missing_values: int = 0
    duplicate_rows: int = 0
    memory_mb: float = 0.0
    label_column: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    unidirectional_records: int = 0
    bidirectional_records: int = 0
    mapped_columns: Dict[str, str] = field(default_factory=dict)
    unmapped_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    duplicate_columns_removed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": round(self.memory_mb, 3),
            "label_column": self.label_column,
            "classes": self.classes,
            "unidirectional_records": self.unidirectional_records,
            "bidirectional_records": self.bidirectional_records,
            "mapped_columns": self.mapped_columns,
            "unmapped_columns": self.unmapped_columns,
            "feature_columns": self.feature_columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "duplicate_columns_removed": self.duplicate_columns_removed,
            "warnings": self.warnings,
        }


def normalize_name(name: str) -> str:
    """Normalise a column name for tolerant alias matching."""
    cleaned = str(name).strip().lower().replace("%", "percent")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def find_label_column(columns: Iterable[str]) -> Optional[str]:
    """Return the best matching target column name, if any."""
    normalized = {normalize_name(col): col for col in columns}
    for alias in LABEL_ALIASES:
        if alias in normalized:
            return normalized[alias]
    for alias in LABEL_ALIASES:
        token = alias.replace(" ", "")
        for norm_col, original in normalized.items():
            if norm_col.replace(" ", "") == token:
                return original
    return None


def canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Rename dataset columns to canonical names.

    Returns the renamed frame, the ``original -> canonical`` mapping and the
    list of columns that could not be mapped. Duplicated canonical targets
    (``IP Address`` and ``Src IP`` both mapping to ``source_ip``) keep the first
    occurrence and drop the rest, which is reported to the caller.
    """
    mapping: Dict[str, str] = {}
    unmapped: List[str] = []
    taken: Dict[str, str] = {}
    renamed: Dict[str, str] = {}

    for original in df.columns:
        canonical = LOOKUP.get(normalize_name(original))
        if canonical is None:
            unmapped.append(str(original))
            continue
        if canonical in taken:
            unmapped.append(str(original))
            continue
        taken[canonical] = str(original)
        mapping[str(original)] = canonical
        if str(original) != canonical:
            renamed[str(original)] = canonical

    return df.rename(columns=renamed), mapping, unmapped


def read_dataset(path: str) -> pd.DataFrame:
    """Read a CSV dataset with defensive encoding and separator handling."""
    if not os.path.exists(path):
        raise DatasetError(
            "Dataset not found. Place a CSV file in the 'data' folder "
            "(default: data/traffic_dataset.csv) or upload one from the web UI."
        )
    if os.path.getsize(path) == 0:
        raise DatasetError("The dataset file is empty (0 bytes).")

    frame: Optional[pd.DataFrame] = None
    last_error: Optional[Exception] = None
    attempts = (
        (",", "c"),
        (";", "python"),
        ("\t", "python"),
        ("|", "python"),
        (None, "python"),
    )
    for encoding in ("utf-8", "latin-1"):
        for sep, engine in attempts:
            try:
                frame = pd.read_csv(path, encoding=encoding, sep=sep, engine=engine,
                                    skip_blank_lines=True, low_memory=False)
            except Exception as exc:  # noqa: BLE001 - retried with other separators
                last_error = exc
                continue
            if frame.shape[1] > 1:
                break
        if frame is not None and frame.shape[1] > 1:
            break

    if frame is None or frame.shape[1] <= 1:
        raise DatasetError(
            "The file could not be parsed as a valid CSV table. "
            f"Parser reported: {last_error}"
        )
    frame.columns = [str(col).strip() for col in frame.columns]
    if frame.empty or frame.shape[0] == 0:
        raise DatasetError("The dataset contains a header but no data rows.")
    return frame


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return series.astype("float64")
        except (TypeError, ValueError):
            return series
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_frame(df: pd.DataFrame, label_column: Optional[str] = None) -> pd.DataFrame:
    """Replace infinite values, coerce numeric columns and strip whitespace."""
    frame = df.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str).str.strip()
            frame[column] = frame[column].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    for column in frame.columns:
        if column == label_column:
            continue
        converted = _to_numeric(frame[column])
        if converted.notna().sum() > 0 and converted.notna().sum() >= 0.6 * frame[column].notna().sum():
            frame[column] = converted
    numeric_frame = frame.select_dtypes(include=[np.number])
    if not numeric_frame.empty:
        frame[numeric_frame.columns] = numeric_frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _duration_to_seconds(series: pd.Series) -> pd.Series:
    """Convert flow duration to seconds, auto-detecting microsecond datasets."""
    values = pd.to_numeric(series, errors="coerce")
    positive = values.dropna()
    positive = positive[positive > 0]
    if positive.empty:
        return values / 1000.0
    median = float(positive.median())
    if median > 1e6:
        return values / 1_000_000.0
    if median > 1000:
        return values / 1000.0
    return values


def _subnet(ip_series: pd.Series) -> pd.Series:
    def to_subnet(value: Any) -> str:
        text = str(value)
        if "." in text and text.count(".") == 3:
            parts = text.split(".")
            try:
                return ".".join(parts[:3]) + ".0/24"
            except ValueError:
                return "unknown"
        return "unknown"
    return ip_series.map(to_subnet).replace("nan", "unknown").fillna("unknown")


def _port_span(source_port: pd.Series, destination_port: pd.Series) -> pd.Series:
    """Absolute distance between the two transport-layer ports."""
    return (destination_port - source_port).abs()


def derive_traffic_direction(frame: pd.DataFrame) -> pd.Series:
    """Classify every flow as FORWARD / REVERSE / UNIDIRECTIONAL / UNKNOWN.

    Explicit direction columns win. Otherwise the forward/reverse packet and
    byte counters are compared; a flow with traffic in a single direction only
    is reported as UNIDIRECTIONAL, which is the flow family this project
    monitors. When no reverse counters exist the flow is inferred from the
    source/destination port ordering.
    """
    if "traffic_direction" in frame.columns:
        explicit = frame["traffic_direction"].astype(str).str.strip().str.upper()
        mapped = explicit.replace(
            {
                "FWD": DIRECTION_FORWARD,
                "FORWARD": DIRECTION_FORWARD,
                "OUTBOUND": DIRECTION_FORWARD,
                "OUT": DIRECTION_FORWARD,
                "SRC->DST": DIRECTION_FORWARD,
                "SRC-DST": DIRECTION_FORWARD,
                "S2D": DIRECTION_FORWARD,
                "A->B": DIRECTION_FORWARD,
                "REV": DIRECTION_REVERSE,
                "REVERSE": DIRECTION_REVERSE,
                "REVERSED": DIRECTION_REVERSE,
                "INBOUND": DIRECTION_REVERSE,
                "IN": DIRECTION_REVERSE,
                "DST->SRC": DIRECTION_REVERSE,
                "DST-SRC": DIRECTION_REVERSE,
                "D2S": DIRECTION_REVERSE,
                "B->A": DIRECTION_REVERSE,
                "UNIDIRECTIONAL": DIRECTION_UNIDIRECTIONAL,
                "UNI-DIRECTIONAL": DIRECTION_UNIDIRECTIONAL,
                "ONE WAY": DIRECTION_UNIDIRECTIONAL,
                "ONE-WAY": DIRECTION_UNIDIRECTIONAL,
                "ONEWAY": DIRECTION_UNIDIRECTIONAL,
                "U": DIRECTION_UNIDIRECTIONAL,
            }
        )
        mapped = mapped.where(mapped.isin(DIRECTION_VALUES), DIRECTION_UNKNOWN)
        known = mapped != DIRECTION_UNKNOWN
        derived = _derive_direction_from_counters(frame)
        mapped[~known] = derived[~known]
        return mapped

    return _derive_direction_from_counters(frame)


def _safe_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _derive_direction_from_counters(frame: pd.DataFrame) -> pd.Series:
    forward_packets = _safe_series(frame, "forward_packets")
    reverse_packets = _safe_series(frame, "reverse_packets")
    forward_bytes = _safe_series(frame, "forward_bytes")
    reverse_bytes = _safe_series(frame, "reverse_bytes")

    has_reverse_packets = "reverse_packets" in frame.columns
    has_reverse_bytes = "reverse_bytes" in frame.columns

    if not has_reverse_packets and not has_reverse_bytes and "forward_bytes" in frame.columns:
        total = _safe_series(frame, "byte_count")
        reverse_bytes = (total - forward_bytes).clip(lower=0)
        has_reverse_bytes = True

    if not has_reverse_packets and "forward_packets" in frame.columns and "packet_count" in frame.columns:
        total = _safe_series(frame, "packet_count")
        reverse_packets = (total - forward_packets).clip(lower=0)
        has_reverse_packets = True

    if has_reverse_packets or has_reverse_bytes:
        direction = pd.Series(DIRECTION_UNKNOWN, index=frame.index, dtype=object)
        only_forward = (reverse_packets <= 0) & (forward_packets > 0)
        only_reverse = (forward_packets <= 0) & (reverse_packets > 0)
        forward_dominant = (forward_packets >= reverse_packets) & (forward_packets > 0)
        reverse_dominant = (forward_packets < reverse_packets) & (reverse_packets > 0)

        direction[forward_dominant] = DIRECTION_FORWARD
        direction[reverse_dominant] = DIRECTION_REVERSE
        direction[only_forward] = DIRECTION_UNIDIRECTIONAL
        direction[only_reverse] = DIRECTION_UNIDIRECTIONAL

        unknown = direction == DIRECTION_UNKNOWN
        if unknown.any():
            byte_fwd = forward_bytes > 0
            byte_rev = reverse_bytes > 0
            direction[unknown & byte_fwd & ~byte_rev] = DIRECTION_UNIDIRECTIONAL
            direction[unknown & ~byte_fwd & byte_rev] = DIRECTION_UNIDIRECTIONAL
            direction[unknown & byte_fwd & byte_rev & (forward_bytes >= reverse_bytes)] = DIRECTION_FORWARD
            direction[unknown & byte_fwd & byte_rev & (forward_bytes < reverse_bytes)] = DIRECTION_REVERSE

        still_unknown = direction == DIRECTION_UNKNOWN
        direction[still_unknown] = DIRECTION_UNIDIRECTIONAL
        return direction.astype(str)

    source_port = _safe_series(frame, "source_port")
    destination_port = _safe_series(frame, "destination_port")
    direction = pd.Series(DIRECTION_UNKNOWN, index=frame.index, dtype=object)
    known_ports = (source_port > 0) & (destination_port > 0)
    direction[known_ports & (source_port > destination_port)] = DIRECTION_FORWARD
    direction[known_ports & (source_port < destination_port)] = DIRECTION_REVERSE
    direction[~known_ports] = DIRECTION_UNIDIRECTIONAL
    return direction.astype(str)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction, ratio, rate and port-context features."""
    frame = df.copy()

    direction = derive_traffic_direction(frame)
    frame["traffic_direction"] = direction
    frame["is_unidirectional"] = (direction == DIRECTION_UNIDIRECTIONAL).astype("int64")
    frame["is_bidirectional"] = (direction.isin([DIRECTION_FORWARD, DIRECTION_REVERSE])).astype("int64")

    packets = _safe_series(frame, "packet_count")
    if not "packet_count" in frame.columns and "forward_packets" in frame.columns:
        packets = _safe_series(frame, "forward_packets") + _safe_series(frame, "reverse_packets")
    byte_count = _safe_series(frame, "byte_count")
    if "byte_count" not in frame.columns and "forward_bytes" in frame.columns:
        byte_count = _safe_series(frame, "forward_bytes") + _safe_series(frame, "reverse_bytes")

    forward_packets = _safe_series(frame, "forward_packets")
    reverse_packets = _safe_series(frame, "reverse_packets")
    forward_bytes = _safe_series(frame, "forward_bytes")
    reverse_bytes = _safe_series(frame, "reverse_bytes")

    duration_seconds = (
        _duration_to_seconds(frame["flow_duration"]) if "flow_duration" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )

    def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        den = denominator.replace(0, np.nan)
        result = numerator / den
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    frame["flow_duration_seconds"] = duration_seconds.fillna(0.0)
    frame["bytes_per_packet"] = safe_div(byte_count, packets)
    frame["packets_per_second"] = safe_div(packets, duration_seconds)
    frame["bytes_per_second"] = safe_div(byte_count, duration_seconds)
    frame["inter_arrival_per_packet"] = safe_div(duration_seconds, packets)

    packet_total = (forward_packets + reverse_packets).replace(0, np.nan)
    byte_total = (forward_bytes + reverse_bytes).replace(0, np.nan)
    frame["forward_packets_ratio"] = (forward_packets / packet_total).fillna(1.0).clip(0, 1)
    frame["reverse_packets_ratio"] = (reverse_packets / packet_total).fillna(0.0).clip(0, 1)
    frame["forward_bytes_ratio"] = (forward_bytes / byte_total).fillna(1.0).clip(0, 1)

    if "tcp_flags" in frame.columns:
        flags = frame["tcp_flags"]
        frame["tcp_flag_count"] = flags.map(
            lambda value: len(re.findall(r"[A-Za-z]", str(value))) if not pd.api.types.is_numeric_dtype(flags)
            else float(value)
        ).astype("float64")
    else:
        frame["tcp_flag_count"] = 0.0

    destination_port = _safe_series(frame, "destination_port")
    frame["destination_port_privileged"] = ((destination_port > 0) & (destination_port < 1024)).astype("int64")
    source_port = _safe_series(frame, "source_port")
    pair_span = (destination_port - source_port).abs()
    frame["unique_port_ratio"] = safe_div(pair_span, destination_port)

    if "source_ip" in frame.columns:
        frame["source_subnet"] = _subnet(frame["source_ip"])
    else:
        frame["source_subnet"] = "unknown"

    if "source_port" in frame.columns and "destination_port" in frame.columns:
        frame["port_span"] = _port_span(source_port, destination_port)
    else:
        frame["port_span"] = 0.0

    return frame


def determine_feature_columns(
    df: pd.DataFrame, label_column: Optional[str], reference_features: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """Split usable columns into numeric and categorical feature lists."""
    if reference_features:
        features = [column for column in reference_features if column in df.columns]
        missing = [column for column in reference_features if column not in df.columns]
        if missing:
            raise DatasetError(
                "Uploaded dataset is missing features required by the trained model: "
                + ", ".join(missing[:12])
                + ("..." if len(missing) > 12 else "")
            )
        features = list(reference_features)
    else:
        features = [
            column
            for column in df.columns
            if column != label_column and column not in IDENTIFIER_COLUMNS
        ]
        features = [column for column in features if column not in LABEL_ALIASES]
        if not features:
            raise DatasetError("No usable feature columns were found besides the target label.")

    numeric: List[str] = []
    categorical: List[str] = []
    for column in features:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        else:
            converted = _to_numeric(series)
            non_null = series.notna().sum()
            if non_null and converted.notna().sum() >= 0.9 * non_null:
                numeric.append(column)
            else:
                categorical.append(column)
    if not numeric and not categorical:
        raise DatasetError("No numeric or categorical feature columns could be identified.")
    return features, numeric, categorical


def normalise_label(value: Any) -> str:
    """Normalise a raw label value into a readable class name."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return "unknown"
    return text


def is_normal_class(label: Any) -> bool:
    text = normalise_label(label).lower().replace("_", " ").replace("-", " ")
    for token in NORMAL_LABEL_TOKENS:
        if text == token or text.startswith(token + " "):
            return True
    return False


def binary_target(labels: Sequence[Any]) -> np.ndarray:
    """Map class names to 1 (attack) / 0 (normal)."""
    return np.array([0 if is_normal_class(label) else 1 for label in labels], dtype="int64")


def profile_dataset(df: pd.DataFrame, label_column: Optional[str] = None,
                    drop_identifiers: bool = True) -> DatasetReport:
    """Build a :class:`DatasetReport` describing a raw (already canonicalised) frame."""
    frame = df
    missing_before = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    memory_mb = frame.memory_usage(deep=True).sum() / (1024 * 1024)

    report = DatasetReport(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        missing_values=missing_before,
        duplicate_rows=duplicates,
        memory_mb=memory_mb,
        label_column=label_column,
    )

    if "traffic_direction" in frame.columns or any(
        column in frame.columns for column in ("forward_packets", "reverse_packets", "forward_bytes")
    ):
        direction = derive_traffic_direction(frame)
        report.unidirectional_records = int((direction == DIRECTION_UNIDIRECTIONAL).sum())
        report.bidirectional_records = int(direction.isin([DIRECTION_FORWARD, DIRECTION_REVERSE]).sum())

    if label_column and label_column in frame.columns:
        report.classes = sorted({normalise_label(value) for value in frame[label_column].dropna().unique()})

    try:
        features, numeric, categorical = determine_feature_columns(frame, label_column)
        report.feature_columns = features
        report.numeric_features = numeric
        report.categorical_features = categorical
        if not numeric:
            report.warnings.append("No numeric feature columns were detected.")
        if drop_identifiers and not report.numeric_features:
            report.warnings.append("Only categorical features were detected.")
    except DatasetError as exc:
        report.warnings.append(str(exc))

    return report