"""Flask application for AI-based detection of cyber threats in unidirectional IP traffic."""

from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from training import preprocess as pp
from utils.prediction import ModelNotAvailableError, get_predictor, reload_predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DATA_DIR = os.environ.get("APP_DATA_DIR", BASE_DIR)
INSTANCE_DIR = os.path.join(APP_DATA_DIR, "database")
UPLOAD_DIR = os.path.join(APP_DATA_DIR, "uploads")
REPORT_DIR = os.path.join(APP_DATA_DIR, "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DATASET = os.path.join(BASE_DIR, "data", "traffic_dataset.csv")

ALLOWED_EXTENSIONS = {"csv"}
SAMPLE_MARKER = "sample_dataset.csv"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "64"))
MAX_STORED_PREDICTIONS = int(os.environ.get("MAX_STORED_PREDICTIONS", "20000"))
LOGIN_MAX_ATTEMPTS = int(os.environ.get("LOGIN_MAX_ATTEMPTS", "8"))
LOGIN_LOCKOUT_SECONDS = int(os.environ.get("LOGIN_LOCKOUT_SECONDS", "300"))

for directory in (INSTANCE_DIR, UPLOAD_DIR, REPORT_DIR):
    os.makedirs(directory, exist_ok=True)


def utcnow() -> datetime:
    """Naive UTC timestamp, kept explicit for SQLite DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _secret_key() -> str:
    """Read the Flask secret key from the environment, never from source."""
    key = os.environ.get("CYBER_THREAT_SECRET_KEY", "").strip()
    if key:
        return key
    return secrets.token_hex(32)


def _auth_password_hash() -> Optional[str]:
    """Return the stored password hash, or ``None`` when auth is disabled.

    ``CYBER_THREAT_PASSWORD_HASH`` holds a Werkzeug hash. ``CYBER_THREAT_PASSWORD``
    is accepted for convenience and hashed at start-up; it never leaves the process.
    """
    digest = os.environ.get("CYBER_THREAT_PASSWORD_HASH", "").strip()
    if digest:
        return digest
    password = os.environ.get("CYBER_THREAT_PASSWORD", "")
    if not password:
        return None
    return generate_password_hash(password)


PASSWORD_HASH = _auth_password_hash()
AUTH_ENABLED = PASSWORD_HASH is not None
SECURE_COOKIES = os.environ.get("FORCE_SECURE_COOKIES", "0") == "1"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=_secret_key(),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(INSTANCE_DIR, 'cyber_threat.db')}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    JSON_SORT_KEYS=False,
    UPLOAD_FOLDER=UPLOAD_DIR,
    REPORT_FOLDER=REPORT_DIR,
    SAMPLE_DATASET=SAMPLE_DATASET,
    AUTH_ENABLED=AUTH_ENABLED,
    PASSWORD_HASH=PASSWORD_HASH,
    SECURE_COOKIES=SECURE_COOKIES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SECURE_COOKIES,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=int(os.environ.get("SESSION_HOURS", "8"))),
)
_app_password_hash = PASSWORD_HASH
_login_attempts: Dict[str, List[float]] = {}
PUBLIC_ENDPOINTS = frozenset({"login", "login_submit", "logout", "static", "api_health", "favicon"})

db = SQLAlchemy(app)


class Upload(db.Model):
    __tablename__ = "uploads"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    stored_path = db.Column(db.String(512), nullable=True)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    total_records = db.Column(db.Integer, default=0)
    unidirectional_records = db.Column(db.Integer, default=0)
    normal_records = db.Column(db.Integer, default=0)
    malicious_records = db.Column(db.Integer, default=0)
    threat_percentage = db.Column(db.Float, default=0.0)
    model_name = db.Column(db.String(120), default="unknown")
    status = db.Column(db.String(32), default="analysed")
    report_file = db.Column(db.String(255), nullable=True)
    summary_json = db.Column(db.Text, nullable=True)

    predictions = db.relationship("Prediction", backref="upload", lazy=True, cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=False, index=True)
    record_index = db.Column(db.Integer, default=0)
    prediction = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, default=0.0)
    attack_type = db.Column(db.String(64), default="n/a")
    source_ip = db.Column(db.String(64), default="unknown")
    destination_ip = db.Column(db.String(64), default="unknown")
    protocol = db.Column(db.String(32), default="unknown")
    packet_count = db.Column(db.Float, default=0)
    traffic_direction = db.Column(db.String(32), default="UNKNOWN")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.Index("ix_prediction_upload_index", "upload_id", "record_index"),)


class ModelMetric(db.Model):
    __tablename__ = "model_metrics"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(120), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    trained_on = db.DateTime, db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class UserError(Exception):
    """Exception carrying a user facing message."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------------------ security


def auth_enabled() -> bool:
    """Authentication is active when a password hash was supplied by the environment."""
    return bool(app.config.get("AUTH_ENABLED"))


def is_authenticated() -> bool:
    return bool(session.get("authenticated"))


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _validate_csrf() -> bool:
    submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
    expected = session.get("csrf_token", "")
    return bool(expected) and secrets.compare_digest(str(submitted), str(expected))


def _client_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _register_failure() -> None:
    key = _client_key()
    now = time.time()
    history = [stamp for stamp in _login_attempts.get(key, []) if now - stamp < LOGIN_LOCKOUT_SECONDS]
    history.append(now)
    _login_attempts[key] = history


def _locked_out() -> bool:
    key = _client_key()
    now = time.time()
    history = [stamp for stamp in _login_attempts.get(key, []) if now - stamp < LOGIN_LOCKOUT_SECONDS]
    _login_attempts[key] = history
    return len(history) >= LOGIN_MAX_ATTEMPTS


def login_required(view):
    """Protect a single view with the session login when authentication is enabled."""

    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any):
        return view(*args, **kwargs)

    return wrapper


@app.before_request
def enforce_authentication():
    """Single choke point for authentication and CSRF on every request."""
    if not auth_enabled():
        return None
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if not is_authenticated():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required", "status": 401}), 401
        return redirect(url_for("login", next=request.full_path if request.query_string else request.path))
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _validate_csrf():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Invalid or missing CSRF token", "status": 400}), 400
        return render_template("error.html", title="Security Check Failed",
                               message="The form expired or the security token was invalid. "
                                       "Please reload the page and try again.",
                               status=400), 400
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if not auth_enabled():
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        if not _validate_csrf():
            error = "The form expired. Please try again."
        elif _locked_out():
            error = "Too many failed attempts. Try again in a few minutes."
        else:
            password = request.form.get("password", "")
            stored = app.config.get("PASSWORD_HASH") or _app_password_hash
            if stored and check_password_hash(stored, password):
                _login_attempts.pop(_client_key(), None)
                session.clear()
                session["authenticated"] = True
                session["csrf_token"] = secrets.token_urlsafe(32)
                session.permanent = True
                target = request.args.get("next") or request.form.get("next") or ""
                if not target.startswith("/") or target.startswith("//"):
                    target = url_for("index")
                return redirect(target)
            _register_failure()
            error = "Incorrect password."
    return render_template("login.html", title="Sign in", error=error, next=request.args.get("next", "")), (
        401 if error else 200
    )


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login" if auth_enabled() else "index"))


def load_training_metadata() -> Dict[str, Any]:
    path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001 - metadata is advisory only
        return {}


def latest_analysis() -> Optional[Upload]:
    return db.session.query(Upload).filter_by(status="analysed").order_by(Upload.id.desc()).first()


def store_metrics(metadata: Dict[str, Any]) -> None:
    """Mirror the training metrics into SQLite for historical comparison."""
    if not metadata.get("models"):
        return
    db.session.query(ModelMetric).delete()
    trained_on = _parse_timestamp(metadata.get("generated_at"))
    for record in metadata["models"]:
        db.session.add(
            ModelMetric(
                model_name=str(record.get("model_name", "unknown")),
                accuracy=record.get("accuracy"),
                precision=record.get("precision"),
                recall=record.get("recall"),
                f1_score=record.get("f1_score"),
                roc_auc=record.get("roc_auc"),
                trained_on=trained_on,
            )
        )
    db.session.commit()


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return utcnow()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return utcnow()


def save_uploaded_file(file_storage) -> Tuple[str, str]:
    """Validate and persist an uploaded CSV using a sanitised filename."""
    if file_storage is None or not file_storage.filename:
        raise UserError("No file was selected. Please choose a CSV file and try again.")
    original = secure_filename(file_storage.filename)
    if not allowed_file(original):
        raise UserError(
            "Invalid file type. Only .csv datasets are accepted "
            f"(received: {file_storage.filename.rsplit('.', 1)[-1] if '.' in file_storage.filename else 'unknown'})."
        )
    stored_name = f"{utcnow().strftime('%Y%m%d%H%M%S%f')}_{original}"
    stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file_storage.save(stored_path)
    if os.path.getsize(stored_path) == 0:
        os.remove(stored_path)
        raise UserError("The uploaded file is empty. Please provide a CSV file with at least one data row.")
    return original, stored_path


def analyse_file(path: str, original_name: str) -> Upload:
    """Run the prediction pipeline and persist the results."""
    predictor = get_predictor()
    if not predictor.available:
        raise UserError(predictor.error or "No trained model is available.", status=503)

    result = predictor.analyse_csv(path)
    if result.total_records == 0:
        raise UserError("The uploaded dataset does not contain any traffic records.")

    upload = Upload(
        filename=original_name,
        stored_path=path,
        total_records=result.total_records,
        unidirectional_records=result.unidirectional_records,
        normal_records=result.normal_records,
        malicious_records=result.malicious_records,
        threat_percentage=round(result.threat_percentage, 2),
        model_name=result.model_name,
        status="analysed",
        summary_json=json.dumps(result.summary_dict()),
    )
    db.session.add(upload)
    db.session.flush()

    records = result.records_frame
    limit = min(len(records), MAX_STORED_PREDICTIONS)
    now = utcnow()
    rows = [
        {
            "upload_id": upload.id,
            "record_index": index,
            "prediction": str(row["Prediction"]),
            "probability": float(row["Threat Probability"]),
            "attack_type": str(row["Attack Type"]),
            "source_ip": str(row["Source IP"])[:64],
            "destination_ip": str(row["Destination IP"])[:64],
            "protocol": str(row["Protocol"])[:32],
            "packet_count": float(row["Packet Count"]),
            "traffic_direction": str(row["Traffic Direction"])[:32],
            "timestamp": now,
        }
        for index, (_, row) in enumerate(records.head(limit).iterrows())
    ]
    if rows:
        db.session.bulk_insert_mappings(Prediction, rows)

    report_name = f"predictions_upload_{upload.id}.csv"
    result.records_frame.to_csv(os.path.join(REPORT_DIR, report_name), index=False)
    upload.report_file = report_name

    db.session.commit()
    return upload


def build_chart_payload(upload: Optional[Upload], summary: Dict[str, Any], metadata: Dict[str, Any],
                        frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Assemble every value needed by the Chart.js dashboards."""
    models = [
        {
            "model_name": record.get("model_name"),
            "accuracy": record.get("accuracy"),
            "precision": record.get("precision"),
            "recall": record.get("recall"),
            "f1_score": record.get("f1_score"),
            "roc_auc": record.get("roc_auc"),
        }
        for record in metadata.get("models", [])
    ]

    if upload is None:
        total = int(metadata.get("train_records", 0) or 0) + int(metadata.get("test_records", 0) or 0)
        normal = int(metadata.get("normal_records", 0) or 0)
        malicious = max(total - normal, 0)
        unidirectional = int(metadata.get("unidirectional_records", 0) or 0)
        bidirectional = int(metadata.get("bidirectional_records", 0) or 0)
        attack_distribution = metadata.get("attack_distribution", {}) or {}
        normal_label = next(
            (name for name in attack_distribution if pp.is_normal_class(name)), None
        )
        attack_types = {name: count for name, count in attack_distribution.items() if name != normal_label}
        return {
            "source": "training",
            "total_records": total,
            "normal": normal,
            "malicious": malicious,
            "threat_percentage": round(100.0 * malicious / total, 2) if total else 0.0,
            "unidirectional": unidirectional,
            "bidirectional": bidirectional,
            "attack_types": attack_types,
            "protocols": {},
            "directions": {
                pp.DIRECTION_UNIDIRECTIONAL: unidirectional,
                "BIDIRECTIONAL": bidirectional,
            },
            "probability_histogram": {},
            "top_sources": [],
            "confusion_matrix": None,
            "models": models,
            "table": {"columns": [], "rows": []},
        }

    protocols = summary.get("protocol_distribution", {}) or {}
    table_columns = ["Source IP", "Destination IP", "Protocol", "Packet Count",
                     "Traffic Direction", "Prediction", "Threat Probability", "Attack Type"]
    rows: List[Dict[str, Any]] = []
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        rows = frame.head(25).to_dict(orient="records")

    return {
        "source": "upload",
        "total_records": summary.get("total_records", upload.total_records),
        "normal": summary.get("normal", upload.normal_records),
        "malicious": summary.get("malicious", upload.malicious_records),
        "threat_percentage": summary.get("threat_percentage", upload.threat_percentage),
        "unidirectional": summary.get("unidirectional_records", upload.unidirectional_records),
        "bidirectional": summary.get("bidirectional_records", 0),
        "attack_types": summary.get("attack_types", {}) or {},
        "protocols": protocols,
        "directions": summary.get("direction_distribution", {}) or {},
        "probability_histogram": summary.get("probability_histogram", {}) or {},
        "top_sources": summary.get("top_sources", []) or [],
        "confusion_matrix": summary.get("confusion_matrix"),
        "ground_truth_accuracy": summary.get("ground_truth_accuracy"),
        "model_name": summary.get("model_name", upload.model_name),
        "models": models,
        "table": {"columns": table_columns, "rows": rows},
    }


def dashboard_context(upload: Optional[Upload]) -> Dict[str, Any]:
    """Build the dictionary consumed by the dashboard and results templates."""
    metadata = load_training_metadata()
    sample_available = os.path.exists(app.config["SAMPLE_DATASET"])
    if upload is None:
        return {
            "source": "training",
            "has_upload": False,
            "metadata": metadata,
            "summary": {},
            "upload": None,
            "chart_payload": build_chart_payload(None, {}, metadata),
            "recent_uploads": recent_uploads(),
            "traffic_dataset_available": sample_available,
        }

    summary: Dict[str, Any] = {}
    if upload.summary_json:
        try:
            summary = json.loads(upload.summary_json)
        except json.JSONDecodeError:
            summary = {}

    frame = (
        db.session.query(Prediction)
        .filter(Prediction.upload_id == upload.id)
        .order_by(Prediction.record_index)
        .limit(250)
        .all()
    )
    frame_df = pd.DataFrame(
        [
            {
                "Source IP": record.source_ip,
                "Destination IP": record.destination_ip,
                "Protocol": record.protocol,
                "Packet Count": record.packet_count,
                "Traffic Direction": record.traffic_direction,
                "Prediction": record.prediction,
                "Threat Probability": round(float(record.probability), 2),
                "Attack Type": record.attack_type,
            }
            for record in frame
        ]
    )

    payload = build_chart_payload(upload, summary, metadata, frame_df)
    return {
        "source": "upload",
        "has_upload": True,
        "metadata": metadata,
        "summary": summary,
        "upload": upload,
        "predictions": frame_df,
        "chart_payload": payload,
        "recent_uploads": recent_uploads(),
        "traffic_dataset_available": sample_available,
    }


def recent_uploads(limit: int = 8) -> List[Upload]:
    return (
        db.session.query(Upload)
        .filter_by(status="analysed")
        .order_by(Upload.upload_time.desc(), Upload.id.desc())
        .limit(limit)
        .all()
    )


def history_rows() -> List[Dict[str, Any]]:
    rows = (
        db.session.query(
            Upload.id,
            Upload.filename,
            Upload.upload_time,
            Upload.total_records,
            Upload.unidirectional_records,
            Upload.normal_records,
            Upload.malicious_records,
            Upload.threat_percentage,
            Upload.model_name,
        )
        .order_by(Upload.id.desc())
        .limit(25)
        .all()
    )
    keys = (
        "id", "filename", "upload_time", "total_records", "unidirectional_records",
        "normal_records", "malicious_records", "threat_percentage", "model_name",
    )
    return [dict(zip(keys, row)) for row in rows]


@app.context_processor
def inject_globals() -> Dict[str, Any]:
    predictor = get_predictor()
    metadata = load_training_metadata()
    return {
        "model_available": predictor.available,
        "model_error": predictor.error,
        "best_model": metadata.get("best_model", "not trained"),
        "training_metadata": metadata,
        "nav_history": history_rows(),
        "current_year": utcnow().year,
        "csrf_token": csrf_token,
        "auth_enabled": auth_enabled(),
        "is_authenticated": is_authenticated,
    }


@app.route("/")
def index():
    predictor = get_predictor()
    metadata = load_training_metadata()
    stats = {
        "models": len(metadata.get("models", [])),
        "features": len(metadata.get("feature_columns", [])),
        "records": metadata.get("train_records", 0) + metadata.get("test_records", 0),
        "classes": len(metadata.get("class_names", [])),
    }
    return render_template(
        "index.html",
        title="Home",
        stats=stats,
        model_available=predictor.available,
        model_error=predictor.error,
        metadata=metadata,
        uploads=recent_uploads(),
    )


@app.route("/dashboard")
def dashboard():
    upload = latest_analysis()
    context = dashboard_context(upload)
    return render_template("dashboard.html", title="Dashboard", **context)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    preview: Optional[Dict[str, Any]] = None
    stored_filename = ""
    error = None

    if request.method == "POST":
        try:
            original, stored_path = save_uploaded_file(request.files.get("dataset"))
            raw = pp.read_dataset(stored_path)
            canonical, mapping, unmapped = pp.canonicalize_columns(raw)
            label_column = pp.find_label_column(canonical.columns)
            report = pp.profile_dataset(canonical, label_column)
            preview = report.to_dict()
            preview["mapped_columns"] = mapping
            preview["unmapped_columns"] = unmapped
            preview["label_classes"] = report.classes
            stored_filename = os.path.basename(stored_path)
        except UserError as exc:
            error = exc.message
        except pp.DatasetError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - user facing message only
            error = f"The dataset could not be read: {exc}"

    return render_template(
        "upload.html",
        title="Upload Dataset",
        preview=preview,
        stored_filename=stored_filename,
        error=error,
        sample_available=os.path.exists(app.config["SAMPLE_DATASET"]),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file_storage = request.files.get("dataset")
        if file_storage is not None and file_storage.filename:
            original, stored_path = save_uploaded_file(file_storage)
        else:
            stored_name = secure_filename(request.form.get("stored_filename", ""))
            if stored_name == SAMPLE_MARKER:
                if not os.path.exists(app.config["SAMPLE_DATASET"]):
                    raise UserError(
                        "No bundled sample dataset was found at data/traffic_dataset.csv. "
                        "Generate one with 'python data/make_sample_dataset.py' or upload your own CSV."
                    )
                stored_path = app.config["SAMPLE_DATASET"]
                original = "traffic_dataset.csv (bundled sample)"
            else:
                candidate = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
                if not stored_name or not os.path.exists(candidate):
                    raise UserError("The previously uploaded file is no longer available. Please upload it again.")
                stored_path = candidate
                original = request.form.get("original_filename") or stored_name

        upload = analyse_file(stored_path, original)
        return redirect(url_for("results", upload_id=upload.id))
    except UserError as exc:
        return render_template("error.html", title="Prediction Error", message=exc.message,
                               status=exc.status), exc.status
    except ModelNotAvailableError as exc:
        return render_template("error.html", title="Model Not Available", message=str(exc), status=503), 503
    except pp.DatasetError as exc:
        return render_template("error.html", title="Invalid Dataset", message=str(exc), status=400), 400
    except Exception as exc:  # noqa: BLE001 - user facing message only
        return render_template("error.html", title="Prediction Error",
                               message=f"Prediction failed: {exc}", status=500), 500


@app.route("/results/<int:upload_id>")
def results(upload_id: int):
    upload = db.session.get(Upload, upload_id)
    if upload is None:
        return render_template(
            "error.html", title="Not Found",
            message="This analysis does not exist. Upload a dataset to create a new one.",
            status=404,
        ), 404
    context = dashboard_context(upload)
    context["title"] = "Detection Results"
    return render_template("results.html", **context)


@app.route("/metrics")
def metrics():
    metadata = load_training_metadata()
    stored = (
        db.session.query(ModelMetric)
        .order_by(ModelMetric.id)
        .all()
    )
    history = [
        {
            "model_name": row.model_name,
            "accuracy": row.accuracy,
            "precision": row.precision,
            "recall": row.recall,
            "f1_score": row.f1_score,
            "roc_auc": row.roc_auc,
        }
        for row in stored
    ]
    return render_template(
        "metrics.html",
        title="Model Performance",
        metadata=metadata,
        stored_metrics=history,
        metadata_ready=bool(metadata.get("models")),
    )


@app.route("/about")
def about():
    metadata = load_training_metadata()
    return render_template(
        "about.html",
        title="About",
        metadata=metadata,
        api_examples={
            "curl": (
                "curl -X POST -F \"dataset=@data/traffic_dataset.csv\" "
                "http://127.0.0.1:5000/api/predict"
            ),
            "response": json.dumps(
                {
                    "total_records": 1000,
                    "normal": 720,
                    "malicious": 280,
                    "threat_percentage": 28.0,
                },
                indent=2,
            ),
        },
    )


@app.route("/live")
def live():
    upload = latest_analysis()
    return render_template("live.html", title="Live Simulation", upload=upload, uploads=recent_uploads())


@app.route("/download/<path:filename>")
def download(filename: str):
    safe_name = secure_filename(filename)
    if safe_name != filename or not allowed_file(safe_name):
        raise UserError("Invalid file name.")
    if not os.path.exists(os.path.join(REPORT_DIR, safe_name)):
        raise UserError("The requested report is not available.")
    return send_from_directory(REPORT_DIR, safe_name, as_attachment=True)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        file_storage = request.files.get("dataset") or request.files.get("file")
        if file_storage is None or not file_storage.filename:
            raise UserError("Attach a CSV file using the 'dataset' form field.")
        _, stored_path = save_uploaded_file(file_storage)
        predictor = get_predictor()
        if not predictor.available:
            raise UserError(predictor.error or "No trained model is available.", status=503)
        result = predictor.analyse_csv(stored_path)
        payload = result.summary_dict()
        payload["predictions_sample"] = result.display_table.head(20).to_dict(orient="records")
        payload["model_available"] = True
        return jsonify(payload), 200
    except UserError as exc:
        return jsonify({"error": exc.message, "status": exc.status}), exc.status
    except pp.DatasetError as exc:
        return jsonify({"error": str(exc), "status": 400}), 400
    except ModelNotAvailableError as exc:
        return jsonify({"error": str(exc), "status": 503}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}", "status": 500}), 500


@app.route("/api/simulation/<int:upload_id>")
def api_simulation(upload_id: int):
    upload = db.session.get(Upload, upload_id)
    if upload is None:
        return jsonify({"error": "Analysis not found", "records": []}), 404
    try:
        limit = max(1, min(int(request.args.get("limit", 8)), 100))
    except (TypeError, ValueError):
        limit = 8
    records = (
        db.session.query(Prediction)
        .filter(Prediction.upload_id == upload_id)
        .order_by(func.random())
        .limit(limit)
        .all()
    )
    return jsonify(
        {
            "upload_id": upload_id,
            "filename": upload.filename,
            "total_records": upload.total_records,
            "records": [
                {
                    "source_ip": record.source_ip,
                    "destination_ip": record.destination_ip,
                    "protocol": record.protocol,
                    "packet_count": record.packet_count,
                    "traffic_direction": record.traffic_direction,
                    "prediction": record.prediction,
                    "threat_probability": round(float(record.probability), 2),
                    "attack_type": record.attack_type,
                }
                for record in records
            ],
        }
    )


@app.route("/api/sensor/sample", methods=["POST"])
def api_sensor_sample():
    """Accept live flow records from ``utils/live_sensor.py`` and score them.

    The sensor runs on the operator's machine and pushes flow metadata only;
    the server never captures packets itself.
    """
    if not app.config.get("SENSOR_API_ENABLED", True):
        return jsonify({"error": "Sensor ingestion is disabled on this server", "status": 403}), 403

    payload = request.get_json(silent=True) or {}
    records = payload.get("records") if isinstance(payload, dict) else None
    if not records or not isinstance(records, list):
        return jsonify({"error": "Send a JSON object with a non-empty 'records' list", "status": 400}), 400
    if len(records) > int(MAX_STORED_PREDICTIONS):
        records = records[: int(MAX_STORED_PREDICTIONS)]

    predictor = get_predictor()
    if not predictor.available:
        return jsonify({"error": predictor.error or "No trained model is available", "status": 503}), 503

    try:
        frame = pd.DataFrame(records)
        canonical, _, unmapped = pp.canonicalize_columns(frame)
        label_column = pp.find_label_column(canonical.columns)
        canonical = pp.clean_frame(canonical, label_column)
        canonical["Label_normalised"] = "unlabelled"
        canonical = pp.add_engineered_features(canonical)
        result = predictor.predict_frame(canonical)
    except pp.DatasetError as exc:
        return jsonify({"error": str(exc), "status": 400}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Sensor ingestion failed: {exc}", "status": 500}), 500

    interface = str(payload.get("interface", "live sensor"))[:64]
    upload = Upload(
        filename=f"live-sensor::{interface}",
        stored_path=None,
        total_records=result.total_records,
        unidirectional_records=result.unidirectional_records,
        normal_records=result.normal_records,
        malicious_records=result.malicious_records,
        threat_percentage=round(result.threat_percentage, 2),
        model_name=result.model_name,
        status="analysed",
        summary_json=json.dumps(result.summary_dict()),
    )
    db.session.add(upload)
    db.session.flush()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        {
            "upload_id": upload.id,
            "record_index": index,
            "prediction": str(row["Prediction"]),
            "probability": float(row["Threat Probability"]),
            "attack_type": str(row["Attack Type"]),
            "source_ip": str(row["Source IP"])[:64],
            "destination_ip": str(row["Destination IP"])[:64],
            "protocol": str(row["Protocol"])[:32],
            "packet_count": float(row["Packet Count"]),
            "traffic_direction": str(row["Traffic Direction"])[:32],
            "timestamp": now,
        }
        for index, (_, row) in enumerate(result.records_frame.iterrows())
    ]
    if rows:
        db.session.bulk_insert_mappings(Prediction, rows)
    db.session.commit()

    body = result.summary_dict()
    body["analysis_id"] = upload.id
    body["unmapped_columns"] = unmapped
    return jsonify(body), 200


@app.route("/api/health")
def api_health():
    predictor = get_predictor()
    return jsonify(
        {
            "status": "ok",
            "model_available": predictor.available,
            "model": predictor.metadata.get("best_model") if predictor.metadata else None,
            "uploads": db.session.query(func.count(Upload.id)).scalar(),
            "predictions": db.session.query(func.count(Prediction.id)).scalar(),
        }
    )


@app.route("/api/metrics")
def api_metrics():
    metadata = load_training_metadata()
    return jsonify(
        {
            "best_model": metadata.get("best_model"),
            "generated_at": metadata.get("generated_at"),
            "models": [
                {
                    "model_name": record.get("model_name"),
                    "accuracy": record.get("accuracy"),
                    "precision": record.get("precision"),
                    "recall": record.get("recall"),
                    "f1_score": record.get("f1_score"),
                    "roc_auc": record.get("roc_auc"),
                }
                for record in metadata.get("models", [])
            ],
        }
    )


@app.errorhandler(UserError)
def handle_user_error(error: UserError):
    if request.path.startswith("/api/"):
        return jsonify({"error": error.message, "status": error.status}), error.status
    return render_template("error.html", title="Request Error", message=error.message,
                           status=error.status), error.status


@app.errorhandler(413)
def handle_too_large(_error):
    message = (
        f"The uploaded file is larger than the {MAX_UPLOAD_MB} MB limit. "
        "Please upload a smaller CSV sample."
    )
    if request.path.startswith("/api/"):
        return jsonify({"error": message, "status": 413}), 413
    return render_template("error.html", title="File Too Large", message=message, status=413), 413


@app.errorhandler(404)
def handle_not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found", "status": 404}), 404
    return render_template("error.html", title="Page Not Found",
                           message="The requested page does not exist.", status=404), 404


@app.errorhandler(500)
def handle_server_error(_error):
    db.session.rollback()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error", "status": 500}), 500
    return render_template(
        "error.html",
        title="Server Error",
        message="An unexpected error occurred while processing the request.",
        status=500,
    ), 500


@app.cli.command("init-db")
def init_db() -> None:
    """Create the SQLite schema and mirror the training metrics."""
    db.create_all()
    store_metrics(load_training_metadata())
    print("Database initialised at database/cyber_threat.db")


@app.cli.command("reload-model")
def reload_model() -> None:
    """Reload the trained models without restarting the server."""
    predictor = reload_predictor()
    print("Model reloaded:", predictor.metadata.get("best_model") if predictor.metadata else predictor.error)


with app.app_context():
    db.create_all()
    store_metrics(load_training_metadata())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)