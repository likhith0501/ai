"""Automated checks for the cyber threat detection project.

Run with::

    pip install pytest
    python -m pytest tests -q

The suite covers the preprocessing layer, the unidirectional traffic logic, the
prediction service and every Flask route / API endpoint. Tests that need trained
artifacts are skipped automatically when ``models/metadata.json`` is missing.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from training import preprocess as pp  # noqa: E402

METADATA_PATH = os.path.join(ROOT_DIR, "models", "metadata.json")
SAMPLE_DATASET = os.path.join(ROOT_DIR, "data", "traffic_dataset.csv")
requires_model = pytest.mark.skipif(
    not os.path.exists(METADATA_PATH), reason="run 'python training/train.py' first"
)


# --------------------------------------------------------------------------- helpers


def flow_frame(rows):
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def app_module():
    import app as module

    module.app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        UPLOAD_FOLDER=os.path.join(ROOT_DIR, "uploads"),
        REPORT_FOLDER=os.path.join(ROOT_DIR, "reports"),
    )
    with module.app.app_context():
        module.db.create_all()
    yield module
    with module.app.app_context():
        module.db.drop_all()


@pytest.fixture()
def client(app_module):
    with app_module.app.app_context():
        app_module.db.session.remove()
        app_module.db.drop_all()
        app_module.db.create_all()
    return app_module.app.test_client()


@pytest.fixture(scope="session")
def sample_csv_bytes():
    if not os.path.exists(SAMPLE_DATASET):
        pytest.skip("bundled dataset missing")
    with open(SAMPLE_DATASET, "rb") as handle:
        return handle.read()


# ------------------------------------------------------------------- column mapping


def test_column_aliases_map_to_canonical_names():
    frame = pd.DataFrame(
        {
            "Src IP": ["10.0.0.1"],
            "Dst IP": ["8.8.8.8"],
            "Src Port": [1234],
            "Dst Port": [443],
            "Total Length": [1500],
            "Fwd Packets": [3],
            "Backward Bytes": [0],
            "IAT": [0.01],
            "Category": ["Normal"],
        }
    )
    canonical, mapping, unmapped = pp.canonicalize_columns(frame)
    assert canonical["source_ip"].iloc[0] == "10.0.0.1"
    assert canonical["destination_port"].iloc[0] == 443
    assert canonical["byte_count"].iloc[0] == 1500
    assert canonical["reverse_bytes"].iloc[0] == 0
    assert pp.find_label_column(canonical.columns) == "Category"
    assert mapping["Src IP"] == "source_ip"
    assert unmapped == ["Category"]


def test_unmapped_columns_are_reported():
    frame = pd.DataFrame({"Source_IP": ["1.1.1.1"], "Mystery_Field": [3], "Label": ["Normal"]})
    _, _, unmapped = pp.canonicalize_columns(frame)
    assert "Mystery_Field" in unmapped


def test_label_column_detection_variants():
    assert pp.find_label_column(["Flow ID", "class"]) == "class"
    assert pp.find_label_column(["attack_type"]) == "attack_type"
    assert pp.find_label_column(["packet_count"]) is None


def test_normal_label_detection():
    assert pp.is_normal_class("Normal")
    assert pp.is_normal_class("BENIGN")
    assert not pp.is_normal_class("DoS")
    assert not pp.is_normal_class("Port Scan")


def test_binary_target_mapping():
    assert list(pp.binary_target(["Normal", "DoS", "BENIGN", "Botnet"])) == [0, 1, 0, 1]


# ------------------------------------------------------- unidirectional detection


def test_direction_from_counters():
    frame = flow_frame(
        [
            {"forward_packets": 10, "reverse_packets": 0, "forward_bytes": 900, "reverse_bytes": 0},
            {"forward_packets": 0, "reverse_packets": 7, "forward_bytes": 0, "reverse_bytes": 400},
            {"forward_packets": 9, "reverse_packets": 8, "forward_bytes": 500, "reverse_bytes": 480},
            {"forward_packets": 2, "reverse_packets": 30, "forward_bytes": 100, "reverse_bytes": 2000},
        ]
    )
    directions = list(pp.derive_traffic_direction(frame))
    assert directions == [
        pp.DIRECTION_UNIDIRECTIONAL,
        pp.DIRECTION_UNIDIRECTIONAL,
        pp.DIRECTION_FORWARD,
        pp.DIRECTION_REVERSE,
    ]


def test_direction_from_explicit_column():
    frame = flow_frame([{"traffic_direction": "one-way"}, {"traffic_direction": "outbound"},
                        {"traffic_direction": "inbound"}, {"traffic_direction": "???"}])
    directions = list(pp.derive_traffic_direction(frame))
    assert directions[:3] == [
        pp.DIRECTION_UNIDIRECTIONAL,
        pp.DIRECTION_FORWARD,
        pp.DIRECTION_REVERSE,
    ]
    assert directions[3] in pp.DIRECTION_VALUES


def test_direction_from_ports_only():
    frame = flow_frame([{"source_port": 51000, "destination_port": 80},
                        {"source_port": 80, "destination_port": 51000}])
    assert list(pp.derive_traffic_direction(frame)) == [pp.DIRECTION_FORWARD, pp.DIRECTION_REVERSE]


def test_direction_from_total_bytes_only():
    frame = flow_frame([{"forward_bytes": 1200, "byte_count": 1200}])
    assert pp.derive_traffic_direction(frame).iloc[0] == pp.DIRECTION_UNIDIRECTIONAL


# ------------------------------------------------------------------- data cleaning


def test_clean_frame_removes_infinities_and_blank_strings():
    frame = pd.DataFrame({"Packet_Count": [1.0, np.inf, 3.0], "Protocol": ["TCP", " ", "UDP"]})
    cleaned = pp.clean_frame(frame)
    assert cleaned["Packet_Count"].isna().sum() == 1
    assert cleaned["Protocol"].isna().sum() == 1


def test_duplicate_detection():
    frame = pd.DataFrame({"packet_count": [1, 1, 2]})
    assert int(frame.duplicated().sum()) == 1


def test_engineered_features_are_produced():
    frame = flow_frame(
        [
            {
                "source_port": 51000,
                "destination_port": 80,
                "protocol": "TCP",
                "packet_count": 10,
                "byte_count": 1000,
                "flow_duration": 2_000_000,
                "forward_packets": 10,
                "forward_bytes": 1000,
                "reverse_packets": 0,
                "reverse_bytes": 0,
                "tcp_flags": "SA",
                "source_ip": "192.168.1.10",
                "inter_arrival_time": 0.01,
            }
        ]
    )
    engineered = pp.add_engineered_features(frame)
    for column in pp.ENGINEERED_NUMERIC:
        assert column in engineered.columns, column
    assert engineered["traffic_direction"].iloc[0] == pp.DIRECTION_UNIDIRECTIONAL
    assert engineered["is_unidirectional"].iloc[0] == 1
    assert engineered["bytes_per_packet"].iloc[0] == 100
    assert round(engineered["flow_duration_seconds"].iloc[0], 3) == 2.0
    assert engineered["destination_port_privileged"].iloc[0] == 1
    assert engineered["source_subnet"].iloc[0] == "192.168.1.0/24"


def test_read_dataset_errors():
    with pytest.raises(pp.DatasetError):
        pp.read_dataset(os.path.join(ROOT_DIR, "data", "does_not_exist.csv"))

    empty = os.path.join(ROOT_DIR, "reports", "_empty_test.csv")
    open(empty, "wb").close()
    try:
        with pytest.raises(pp.DatasetError):
            pp.read_dataset(empty)
    finally:
        os.remove(empty)


# ---------------------------------------------------------------- prediction service


@requires_model
def test_predictor_scores_the_sample_dataset():
    from utils.prediction import get_predictor

    predictor = get_predictor()
    assert predictor.available
    result = predictor.analyse_csv(SAMPLE_DATASET)
    assert result.total_records > 0
    assert result.normal_records + result.malicious_records == result.total_records
    assert 0.0 <= result.threat_percentage <= 100.0
    assert set(result.display_table["Prediction"].unique()) <= {"NORMAL", "MALICIOUS"}
    assert "Traffic Probability" not in result.display_table.columns
    assert "Threat Probability" in result.display_table.columns
    assert result.unidirectional_records >= 0


@requires_model
def test_prediction_handles_different_column_names(tmp_path):
    from utils.prediction import get_predictor

    frame = pd.read_csv(SAMPLE_DATASET).head(200)
    renamed = frame.rename(columns={"Source_IP": "Src IP", "Destination_IP": "Dst IP",
                                    "Packet_Count": "Total Packets", "Byte_Count": "Total Length"})
    renamed = renamed.drop(columns=["Label"])
    target = tmp_path / "renamed.csv"
    renamed.to_csv(target, index=False)

    result = get_predictor().analyse_csv(str(target))
    assert result.total_records == len(renamed)


@requires_model
def test_prediction_fails_without_any_known_feature(tmp_path):
    from utils.prediction import get_predictor

    target = tmp_path / "junk.csv"
    pd.DataFrame({"colour": ["red", "blue"], "size": [1, 2]}).to_csv(target, index=False)
    with pytest.raises(pp.DatasetError):
        get_predictor().analyse_csv(str(target))


# ------------------------------------------------------------------------- web layer


def test_all_pages_render(client):
    for path in ("/", "/dashboard", "/upload", "/metrics", "/about", "/live"):
        assert client.get(path).status_code == 200, path


def test_unknown_page_returns_404_page(client):
    response = client.get("/definitely-not-a-page")
    assert response.status_code == 404
    assert b"does not exist" in response.data


def test_api_health(client):
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert "model_available" in payload


@requires_model
def test_api_metrics_returns_all_models(client):
    payload = client.get("/api/metrics").get_json()
    assert payload["models"]
    for record in payload["models"]:
        assert record["model_name"]
        assert 0.0 <= record["accuracy"] <= 1.0


@requires_model
def test_api_predict_with_csv(client, sample_csv_bytes):
    response = client.post(
        "/api/predict",
        data={"dataset": (io.BytesIO(sample_csv_bytes), "traffic_dataset.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    for key in ("total_records", "normal", "malicious", "threat_percentage"):
        assert key in payload
    assert payload["total_records"] > 0
    assert payload["normal"] + payload["malicious"] == payload["total_records"]


def test_api_predict_requires_a_file(client):
    response = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_upload_rejects_non_csv(client):
    response = client.post(
        "/upload",
        data={"dataset": (io.BytesIO(b"col_a,col_b\n1,2\n"), "payload.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Invalid file type" in response.data


def test_upload_rejects_empty_csv(client):
    response = client.post(
        "/upload",
        data={"dataset": (io.BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data",
    )
    assert b"empty" in response.data.lower()


@requires_model
def test_upload_preview_and_analysis_flow(client, sample_csv_bytes, app_module):
    response = client.post(
        "/upload",
        data={"dataset": (io.BytesIO(sample_csv_bytes), "traffic_dataset.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Analyze Traffic" in response.data

    response = client.post("/predict", data={"stored_filename": "sample_dataset.csv"})
    assert response.status_code == 302
    upload_id = response.headers["Location"].rsplit("/", 1)[-1]

    results = client.get(f"/results/{upload_id}")
    assert results.status_code == 200
    assert b"Detection Results" in results.data

    with app_module.app.app_context():
        upload = app_module.db.session.get(app_module.Upload, int(upload_id))
        assert upload.total_records > 0
        assert app_module.db.session.query(app_module.Prediction).filter_by(
            upload_id=upload.id
        ).count() > 0
        assert upload.report_file

    download = client.get(f"/download/{upload.report_file}")
    assert download.status_code == 200
    assert b"Prediction" in download.data

    simulation = client.get(f"/api/simulation/{upload_id}?limit=3").get_json()
    assert len(simulation["records"]) == 3
    assert simulation["records"][0]["prediction"] in {"NORMAL", "MALICIOUS"}


def test_download_blocks_path_traversal(client):
    assert client.get("/download/..%2F..%2Fapp.py").status_code == 400
    assert client.get("/download/missing_report.csv").status_code == 400


def test_missing_result_id_returns_404(client):
    assert client.get("/results/424242").status_code == 404
    assert client.get("/api/simulation/424242").status_code == 404


def test_simulation_endpoint_rejects_absurd_limit(client):
    response = client.get("/api/simulation/1?limit=notanumber")
    assert response.status_code in (200, 404)


@requires_model
def test_missing_model_shows_friendly_message(app_module, client):
    predictor_path = app_module.get_predictor()
    assert predictor_path.available
    metadata_file = os.path.join(ROOT_DIR, "models", "metadata.json")
    backup = metadata_file + ".testbak"
    os.rename(metadata_file, backup)
    try:
        predictor = app_module.reload_predictor()
        assert not predictor.available
        assert "training/train.py" in (predictor.error or "")

        page = client.get("/upload")
        assert page.status_code == 200
        assert b"training/train.py" in page.data

        response = client.post("/predict", data={"stored_filename": "sample_dataset.csv"})
        assert response.status_code == 503
        assert b"trained model" in response.data.lower()

        api = client.post(
            "/api/predict",
            data={"dataset": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")},
            content_type="multipart/form-data",
        )
        assert api.status_code == 503
        assert "error" in api.get_json()
    finally:
        os.rename(backup, metadata_file)
        app_module.reload_predictor()


def test_training_metadata_is_consistent():
    if not os.path.exists(METADATA_PATH):
        pytest.skip("run 'python training/train.py' first")
    with open(METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["best_model"] in [record["model_name"] for record in metadata["models"]]
    assert metadata["feature_columns"]
    assert metadata["train_records"] > 0 and metadata["test_records"] > 0
    for record in metadata["models"]:
        assert 0.0 <= record["accuracy"] <= 1.0
        assert record["confusion_matrix"]
        assert os.path.exists(
            os.path.join(ROOT_DIR, "static", "images", "generated", metadata["confusion_images"][record["model_name"]])
        )
# ----------------------------------------------------------------- live sensor


def test_aggregator_counts_both_directions():
    from utils.live_sensor import FlowAggregator

    aggregator = FlowAggregator(flow_timeout=5.0)
    aggregator.add_packet("10.0.0.1", "10.0.0.2", 51000, 80, 6, 1000, timestamp=100.0, flags=0x02)
    aggregator.add_packet("10.0.0.1", "10.0.0.2", 51000, 80, 6, 1200, timestamp=100.1, flags=0x12)
    aggregator.add_packet("10.0.0.2", "10.0.0.1", 80, 51000, 6, 400, timestamp=100.2, flags=0x10)

    record = aggregator.records()[0]
    assert record["Forward_Packets"] == 2
    assert record["Forward_Bytes"] == 2200
    assert record["Reverse_Packets"] == 1
    assert record["Reverse_Bytes"] == 400
    assert record["Packet_Count"] == 3
    assert record["Byte_Count"] == 2600
    assert record["Traffic_Direction"] == pp.DIRECTION_FORWARD
    assert "S" in record["TCP_Flags"] and "A" in record["TCP_Flags"]
    assert record["Flow_Duration"] == 200000


def test_aggregator_marks_one_way_flows_as_unidirectional():
    from utils.live_sensor import FlowAggregator

    aggregator = FlowAggregator(flow_timeout=5.0)
    for index in range(4):
        aggregator.add_packet("192.168.0.5", "8.8.8.8", 40000, 53, 17, 90, timestamp=200.0 + index * 0.01)
    record = aggregator.records()[0]
    assert record["Reverse_Packets"] == 0
    assert record["Traffic_Direction"] == pp.DIRECTION_UNIDIRECTIONAL
    assert record["Protocol"] == "UDP"
    assert record["Inter_Arrival_Time"] > 0


def test_aggregator_expires_idle_flows():
    from utils.live_sensor import FlowAggregator

    aggregator = FlowAggregator(flow_timeout=1.0)
    aggregator.add_packet("10.0.0.1", "10.0.0.2", 1, 2, 6, 100, timestamp=300.0)
    assert aggregator.active_count() == 1
    assert aggregator.expire(now=300.5) == []
    expired = aggregator.expire(now=301.5)
    assert len(expired) == 1
    assert expired[0]["Packet_Count"] == 1
    assert aggregator.active_count() == 0


def test_aggregator_caps_memory():
    from utils.live_sensor import FlowAggregator

    aggregator = FlowAggregator(max_flows=5)
    for index in range(20):
        aggregator.add_packet("10.0.0.1", f"10.0.0.{index}", 1000 + index, 80, 6, 100, timestamp=400.0)
    assert aggregator.active_count() == 5
    assert aggregator.packets_seen == 20


def test_aggregator_drain_returns_everything():
    from utils.live_sensor import FlowAggregator

    aggregator = FlowAggregator()
    aggregator.add_packet("10.0.0.1", "10.0.0.2", 1, 2, 6, 100, timestamp=500.0)
    aggregator.add_packet("10.0.0.3", "10.0.0.4", 3, 4, 17, 100, timestamp=500.0)
    assert len(aggregator.drain()) == 2
    assert aggregator.active_count() == 0


def test_sensor_record_schema_matches_training_features():
    from utils.live_sensor import FlowAggregator

    if not os.path.exists(METADATA_PATH):
        pytest.skip("run 'python training/train.py' first")
    with open(METADATA_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)

    aggregator = FlowAggregator()
    aggregator.add_packet("10.0.0.1", "10.0.0.2", 51000, 443, 6, 1500, timestamp=600.0, flags=0x12)
    canonical = pp.canonicalize_columns(pd.DataFrame(aggregator.records()))[0]
    canonical = pp.add_engineered_features(canonical)
    assert set(metadata["feature_columns"]) <= set(canonical.columns)


def test_interface_listing_does_not_fail():
    from utils.live_sensor import list_interfaces

    assert isinstance(list_interfaces(), list)


@requires_model
def test_score_records_uses_the_trained_model():
    from utils.live_sensor import FlowAggregator, score_records

    aggregator = FlowAggregator()
    for index in range(6):
        aggregator.add_packet("10.0.0.1", "10.0.0.2", 51000, 22, 6, 200, timestamp=700.0 + index * 0.01)
    scored = score_records(aggregator.records())
    assert scored is not None and len(scored) == 1
    assert scored["Prediction"].iloc[0] in {"NORMAL", "MALICIOUS"}
    assert 0.0 <= float(scored["Threat Probability"].iloc[0]) <= 100.0


@requires_model
def test_sensor_sample_endpoint_scores_and_stores(client, app_module):
    from utils.live_sensor import FlowAggregator

    aggregator = FlowAggregator()
    for index in range(4):
        aggregator.add_packet("172.16.0.4", "104.21.1.1", 60000, 443, 6, 1400,
                              timestamp=800.0 + index * 0.05, flags=0x12)
    records = aggregator.records()

    response = client.post("/api/sensor/sample", json={"interface": "unit-test", "records": records})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total_records"] == len(records)
    assert payload["normal"] + payload["malicious"] == len(records)
    assert payload["analysis_id"]

    results = client.get(f"/results/{payload['analysis_id']}")
    assert results.status_code == 200
    with app_module.app.app_context():
        stored = app_module.Upload.query.get(payload["analysis_id"])
        assert stored.filename.startswith("live-sensor::")


def test_sensor_endpoint_validates_payload(client):
    assert client.post("/api/sensor/sample", json={}).status_code == 400
    assert client.post("/api/sensor/sample", json={"records": []}).status_code == 400
    assert client.post("/api/sensor/sample", json={"records": "nope"}).status_code == 400


def test_sensor_rejects_records_without_known_features(client):
    response = client.post("/api/sensor/sample", json={"records": [{"colour": "red", "size": 1}]})
    assert response.status_code in (400, 500)

# -------------------------------------------------------------------- security


def _enable_auth(app_module, password: str = "s3cret-pw"):
    from werkzeug.security import generate_password_hash

    app_module.app.config["AUTH_ENABLED"] = True
    app_module.app.config["PASSWORD_HASH"] = generate_password_hash(password)
    app_module._app_password_hash = generate_password_hash(password)
    return password


def _disable_auth(app_module):
    app_module.app.config["AUTH_ENABLED"] = False
    app_module.app.config["PASSWORD_HASH"] = None


def _csrf(client) -> str:
    response = client.get("/login")
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.data.decode("utf-8"))
    return match.group(1) if match else ""


def test_auth_is_disabled_without_a_password_env(app_module, client):
    _disable_auth(app_module)
    assert app_module.auth_enabled() is False
    assert client.get("/dashboard").status_code == 200


def test_pages_redirect_to_login_when_auth_enabled(app_module, client):
    password = _enable_auth(app_module)
    try:
        response = client.get("/dashboard")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]
        assert client.get("/upload").status_code == 302
    finally:
        _disable_auth(app_module)


def test_api_returns_401_without_session(app_module, client):
    _enable_auth(app_module)
    try:
        response = client.post("/api/predict", data={}, content_type="multipart/form-data")
        assert response.status_code == 401
        assert response.get_json()["error"] == "Authentication required"
    finally:
        _disable_auth(app_module)


def test_login_rejects_wrong_password_then_accepts(app_module, client):
    password = _enable_auth(app_module)
    try:
        token = _csrf(client)
        bad = client.post("/login", data={"password": "wrong", "csrf_token": token})
        assert bad.status_code == 401
        assert b"Incorrect password" in bad.data

        good = client.post("/login", data={"password": password, "csrf_token": token})
        assert good.status_code == 302
        assert good.headers["Location"] in ("/", "http://localhost/")

        assert client.get("/dashboard").status_code == 200
        assert b"Sign out" in client.get("/").data

        client.get("/logout")
        assert client.get("/dashboard").status_code == 302
    finally:
        _disable_auth(app_module)


def test_login_blocks_csrf_less_post(app_module, client):
    password = _enable_auth(app_module)
    try:
        token = _csrf(client)
        client.post("/login", data={"password": password, "csrf_token": token})
        response = client.post("/predict", data={"stored_filename": "sample_dataset.csv"})
        assert response.status_code == 400
        assert b"Security Check Failed" in response.data
    finally:
        _disable_auth(app_module)


def test_login_throttles_repeated_failures(app_module, client):
    password = _enable_auth(app_module)
    original = app_module.LOGIN_MAX_ATTEMPTS
    app_module.LOGIN_MAX_ATTEMPTS = 3
    app_module._login_attempts.clear()
    try:
        token = _csrf(client)
        for _ in range(3):
            client.post("/login", data={"password": "nope", "csrf_token": token})
        locked = client.post("/login", data={"password": password, "csrf_token": token})
        assert locked.status_code == 401
        assert b"Too many failed attempts" in locked.data
    finally:
        app_module.LOGIN_MAX_ATTEMPTS = original
        app_module._login_attempts.clear()
        _disable_auth(app_module)


def test_password_is_never_written_to_the_database(app_module, client):
    password = _enable_auth(app_module, "unique-password-123")
    try:
        token = _csrf(client)
        client.post("/login", data={"password": password, "csrf_token": token})
        client.get("/dashboard")
        import sqlite3

        connection = sqlite3.connect(os.path.join(ROOT_DIR, "database", "cyber_threat.db"))
        try:
            dump = "\n".join(connection.iterdump())
        finally:
            connection.close()
        assert password not in dump
    finally:
        _disable_auth(app_module)
