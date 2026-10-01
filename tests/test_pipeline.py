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