# AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

> 🛡️ An end-to-end machine-learning system that ingests IP traffic flow records, identifies
> unidirectional communication, classifies every flow as **NORMAL** or **MALICIOUS** with a threat
> probability, and reports the results on a cybersecurity dashboard with a full model evaluation.

---

## Project Overview

This project builds a complete detection system rather than a notebook experiment. A CSV traffic
dataset is uploaded through a web interface, cleaned and mapped automatically, enriched with
engineered features (including a derived `traffic_direction` field), and scored by a trained
scikit-learn pipeline. Every analysis is stored in SQLite, rendered on a dark security dashboard
and can be downloaded as a prediction report or consumed through a JSON API.

The repository contains a synthetic sample dataset so the whole system can be executed locally with
two commands, and it accepts real flow exports (CICIDS2017/2018, UNSW-NB15, custom captures) through
the same column-mapping layer.

---

## Problem Statement

Most intrusion detection systems rely on payload signatures. They fail when:

* the payload is encrypted,
* the traffic is almost entirely **one-way**, so no reply pattern exists for correlation,
* the attacker uses low-and-slow behaviour (beaconing, exfiltration, scanning) that looks like noise
  at the packet level.

Flow-level statistical learning generalises these patterns. Because a large share of intrusion
traffic is unidirectional, explicitly modelling that property gives the classifier a strong,
physically meaningful feature: how much of a flow travelled forward versus backward.

---

## Objectives

1. Accept network traffic datasets in CSV format with flexible column naming.
2. Clean the data (duplicates, missing values, infinite values, wrong types).
3. Detect and report which columns are used for training.
4. Identify unidirectional flows and expose the count on the dashboard.
5. Train and compare several ML models using Accuracy, Precision, Recall, F1 and ROC-AUC.
6. Predict NORMAL / MALICIOUS for every record of a new dataset, with a threat probability.
7. Visualise everything on a responsive cybersecurity dashboard.
8. Persist uploads, predictions and model metrics in SQLite.
9. Expose a JSON prediction endpoint.
10. Handle invalid input and missing models with friendly messages.

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                  Flask app.py               │
   Browser  ───────▶│  routes · validation · security · templates │
   (Bootstrap/      └───────┬──────────────────────────┬───────────┘
    Chart.js)                 │                          │
                    ┌───────▼─────────┐        ┌───────▼───────────────────┐
                    │utils/prediction │        │ Flask-SQLAlchemy          │
                    │  loads models   │        │ database/cyber_threat.db  │
                    └───────┬─────────┘        │ uploads · predictions ·   │
                            │                  │ model_metrics            │
                    ┌───────▼─────────┐        └───────▲───────────────────┘
                    │ training/       │                │
                    │ preprocess.py   │                │
                    │ evaluate.py     │                │
                    └───────┬─────────┘                │
                            │                          │
                    ┌───────▼──────────────────────────▼───────────┐
                    │  CSV dataset → features → trained pipelines │
                    │            models/*.pkl + metadata.json     │
                    └──────────────────────────────────────────────┘
```

**Layers**

| Layer | Files | Responsibility |
| --- | --- | --- |
| Presentation | `templates/`, `static/` | Dark cybersecurity UI, Chart.js visualisations |
| Web | `app.py` | Routes, uploads, security, error handling, JSON API |
| Service | `utils/prediction.py` | Loads artifacts, applies training preprocessing, scores records |
| Training | `training/train.py`, `training/preprocess.py`, `training/evaluate.py` | Training, metrics, plots |
| Data | `data/`, `database/` | Sample dataset generator, SQLite persistence |

---

## Technologies Used

| Category | Technology |
| --- | --- |
| Backend | Python 3.10+, Flask, Flask-SQLAlchemy, SQLAlchemy |
| Machine learning | Scikit-learn, XGBoost (optional at runtime), Joblib |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn (training figures), Chart.js (dashboard) |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5.3 |
| Database | SQLite (`database/cyber_threat.db`) |
| Optional | SHAP (explainability), disabled by default |

---

## Dataset

The bundled `data/traffic_dataset.csv` is **generated**, not scraped: `data/make_sample_dataset.py`
builds a reproducible flow table (`random_state = 42`) with class-conditional statistics, deliberate
class overlap so the metrics are realistic rather than perfect, ~14 % one-way flows, injected
missing values, infinite values and duplicate rows.

| Column | Type | Notes |
| --- | --- | --- |
| `Flow_ID` | string | identifier, dropped before training |
| `Source_IP`, `Destination_IP` | string | used for direction derivation and subnet context, not model inputs |
| `Source_Port`, `Destination_Port` | int | kept as numeric features |
| `Protocol`, `Protocol_Type` | string / int | transport protocol |
| `Packet_Count`, `Byte_Count`, `Packet_Length` | float | volume features |
| `Flow_Duration` | int | microseconds in this dataset, auto-detected |
| `Forward_Packets`, `Forward_Bytes` | float | direction counters |
| `Reverse_Packets`, `Reverse_Bytes` | float | direction counters |
| `TCP_Flags` | string | flag signature |
| `Inter_Arrival_Time` | float | timing behaviour |
| `Label` | string | `Normal`, `DoS`, `DDoS`, `Port Scan`, `Brute Force`, `Botnet`, `Infiltration`, `Malware` |

10 003 rows × 18 columns, 8 classes, 1 521 unidirectional flows.

**Attack labels are dynamic.** Nothing is hard-coded: the label column is discovered, every class
found becomes a category, and classes whose name starts with *normal / benign / legitimate* are
collapsed into the `NORMAL` verdict for the binary detector. Binary datasets
(`Normal` vs `Attack`) and multiclass datasets both work; when more than two classes exist an
additional multiclass model is trained for attack typing.

**Column mapping.** `training/preprocess.py` normalises names (case, punctuation, `%`) and maps
aliases onto canonical features, e.g. `Src IP → source_ip`, `Dst Port → destination_port`,
`Total Length → byte_count`, `Fwd Packets → forward_packets`, `Backward Bytes → reverse_bytes`,
`Fwd Bytes → forward_bytes`, `Avg Packet Size → packet_length`, `Duration → flow_duration`,
`IAT → inter_arrival_time`, `Flags → tcp_flags`, `Proto Type → protocol_type`,
`Category → label`. Columns that cannot be mapped are ignored and listed in the UI. If a feature
expected by the trained model is missing from an upload, it is imputed from training values and a
warning is shown instead of failing silently.

---

## Data Preprocessing

`training/preprocess.py` performs, in order:

1. **Read** with defensive encoding (UTF-8 → Latin-1) and separator detection (`, ; tab |`).
2. **Validate**: file exists, is non-empty, parses as a table and contains rows.
3. **Canonicalise** column names through the alias layer.
4. **Clean**: whitespace strip, numeric coercion of columns that are ≥ 90 % numeric, `±inf → NaN`.
5. **De-duplicate** rows (the bundled dataset contains 3 duplicates that are removed).
6. **Handle missing values** inside the pipeline: `SimpleImputer(median)` for numeric columns and
   `SimpleImputer(most_frequent)` for categorical ones, fitted on training data only.
7. **Encode** categoricals with `OneHotEncoder(handle_unknown="ignore", min_frequency=2)`.
8. **Scale** numerics with `StandardScaler`.
9. **Drop identifiers** (`Flow_ID`, IP addresses) after they have been used for derivation.
10. **Split** stratified 80/20 with `random_state = 42` **before** fitting the preprocessor, so no
    test information leaks into training (276 missing values were handled this way in the run below).

The fitted `ColumnTransformer` is stored inside each model pipeline (`models/*.pkl`) and again on
its own as `models/scaler.pkl`, which guarantees that prediction reuses exactly the same
imputation, encoding and scaling.

---

## Feature Engineering

Beyond the raw flow counters, `add_engineered_features()` creates:

| Feature | Definition |
| --- | --- |
| `traffic_direction` | `FORWARD` / `REVERSE` / `UNIDIRECTIONAL` / `UNKNOWN` |
| `is_unidirectional`, `is_bidirectional` | one-hot flags of the above |
| `bytes_per_packet` | `byte_count / packet_count` |
| `packets_per_second`, `bytes_per_second` | rate features (duration auto-normalised to seconds) |
| `flow_duration_seconds` | duration converted from µs/ms/s by median-value detection |
| `inter_arrival_per_packet` | duration per packet |
| `forward_packets_ratio`, `reverse_packets_ratio`, `forward_bytes_ratio` | direction share |
| `tcp_flag_count` | number of flags in a flag string (or numeric value) |
| `destination_port_privileged` | destination port < 1024 |
| `unique_port_ratio`, `port_span` | port asymmetry |
| `source_subnet` | /24 of the source address (low-cardinality categorical) |

### Unidirectional traffic identification

The logic is a cascade, so it works with very different schemas:

1. If the dataset carries an explicit direction column, its values are normalised
   (`fwd/outbound/src->dst → FORWARD`, `rev/inbound/dst->src → REVERSE`,
   `one-way/uni-directional → UNIDIRECTIONAL`).
2. Otherwise forward/reverse packet counters decide: reverse = 0 → `UNIDIRECTIONAL`,
   forward = 0 → `UNIDIRECTIONAL`, forward ≥ reverse → `FORWARD`, otherwise `REVERSE`.
3. If packet counters are missing, reverse bytes are derived as `byte_count − forward_bytes` and the
   byte counters decide the same way.
4. If only one direction exists at all, the port ordering is used (`src > dst → FORWARD`,
   `src < dst → REVERSE`, otherwise `UNIDIRECTIONAL`).

The dashboard reports the number of unidirectional flows for both the training dataset and every
uploaded analysis.

---

## ML Algorithms

Five classifiers are trained and compared on the same split:

| Model | Purpose |
| --- | --- |
| Logistic Regression | linear, highly interpretable baseline |
| Decision Tree | transparent rule extraction |
| Random Forest (200 trees) | robust bagged ensemble |
| Gradient Boosting (250 stages) | strong sequential ensemble |
| XGBoost (350 trees, hist) | regularised boosting, skipped automatically when unavailable |

### Class imbalance

Attack records are the minority class. The pipeline uses **cost-sensitive learning plus a
stratified split** rather than resampling, so no record is duplicated and the evaluation set stays
representative:

* `class_weight="balanced"` for Logistic Regression, Decision Tree and
  `class_weight="balanced_subsample"` for Random Forest,
* `scale_pos_weight = n_normal / n_attack` for XGBoost,
* `stratify=y` on the train/test split so every class is represented in the hold-out set.

### Model selection

The deployed model is chosen by **0.5·F1 + 0.5·ROC-AUC**, not accuracy alone — accuracy alone
rewards ignoring the minority attack class.

---

## Training Process

```bash
python training/train.py                      # default dataset
python training/train.py --dataset path.csv --test-size 0.25 --random-state 7
```

1. Load and validate the dataset.
2. Canonicalise columns, report mapped/ignored columns.
3. Clean data, drop duplicates, drop unknown label rows.
4. Engineer features including `traffic_direction`.
5. Split X / y (binary `Normal` vs `Attack`) and X / y (multiclass attack type).
6. Stratified hold-out split.
7. Fit the preprocessing pipeline **on the training split only**.
8. Train all models, evaluate each with the full metric bundle.
9. Render Matplotlib/Seaborn confusion matrices, ROC curves and a comparison chart.
10. Persist pipelines with Joblib (`models/<model>.pkl`, `models/best_model.pkl`,
    `models/scaler.pkl`, `models/attack_classifier.pkl`), write `models/metadata.json`,
    `reports/training_report.json` and mirror the metrics into SQLite.

---

## Detection Process

```bash
python -c "from utils.prediction import get_predictor; print(get_predictor().analyse_csv('data/traffic_dataset.csv').summary_dict())"
```

1. Read the CSV exactly like the training step.
2. Canonicalise, clean, engineer the same features.
3. Align to the feature list stored in `models/metadata.json` (absent features are imputed).
4. `predict()` and `predict_proba()` on the stored pipeline — **no retraining, ever**.
5. Threshold the attack probability (default 0.5) to obtain NORMAL / MALICIOUS.
6. Use the multiclass model (when present) to attach an attack type.
7. Aggregate distributions, store per-record rows in SQLite, write the prediction CSV.

Example output:

```
Traffic Record
Prediction: MALICIOUS
Threat Probability: 94.7%
```

---

## System Workflow

```
Dataset Upload
      ↓
Dataset Validation
      ↓
Data Cleaning
      ↓
Feature Extraction
      ↓
Unidirectional Traffic Identification
      ↓
Feature Encoding
      ↓
Feature Scaling
      ↓
ML Model
      ↓
Threat Prediction
      ↓
Threat Probability
      ↓
Dashboard
      ↓
Database Storage
      ↓
Report / CSV Download
```

---

## Dashboard

Dark, responsive cybersecurity interface built with Bootstrap 5 and Chart.js. Cards show
**Total Traffic, Normal Traffic, Malicious Traffic, Unidirectional Traffic** and **Threat Rate**;
charts show normal vs malicious, attack-type distribution, protocol distribution, traffic direction,
threat probability distribution and the model performance comparison. A top-talkers panel, the
prediction history table and links to the live simulation complete the view.

Section highlights: 🛡️ AI Threat Detection · 📊 Traffic Analytics · 🚨 Threat Alerts ·
🤖 ML Models · 📈 Performance Metrics.

---

## Performance Evaluation

Numbers below come from the run stored in `models/metadata.json`
(dataset `traffic_dataset.csv`, 10 000 usable rows, 8 000 train / 2 000 test,
`random_state = 42`). Re-running the training script regenerates them; nothing is hard-coded.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8575 | 0.9374 | 0.8127 | 0.8706 | 0.8795 | 0.2 s |
| Decision Tree | 0.7860 | 0.8113 | 0.8305 | 0.8208 | 0.7763 | 0.4 s |
| Random Forest | 0.8800 | 0.9181 | 0.8746 | 0.8958 | 0.8632 | 2.4 s |
| **Gradient Boosting** (selected) | **0.8885** | **0.9299** | **0.8771** | **0.9027** | **0.8861** | 15.1 s |
| XGBoost | 0.8560 | 0.8954 | 0.8559 | 0.8752 | 0.8757 | 2.2 s |

Attack-type classifier (8 classes, XGBoost multiclass): accuracy **0.8200**, weighted F1
**0.8197**, macro one-vs-rest ROC-AUC **0.9456**.

The sample dataset deliberately contains 25 % label noise and heavy class overlap, so these
figures are conservative; real captures with clearer separation score higher. What matters for the
methodology is that every number is produced by the code at run time.

Top features by importance (Gradient Boosting, impurity based):

| Feature | Importance share |
| --- | --- |
| `packet_count` | 42.0 % |
| `destination_port` | 24.5 % |
| `bytes_per_second` | 13.4 % |
| `inter_arrival_time` | 6.2 % |
| `flow_duration_seconds` | 2.6 % |
| `unique_port_ratio` | 2.0 % |

Confusion matrices and ROC curves for every model are rendered with Matplotlib/Seaborn and shown on
the **Model Performance** page; SHAP is used automatically when installed, otherwise the page falls
back to the estimator's native importance.

---

## Installation

Windows (PowerShell / cmd):

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install shap        # enables SHAP explanations
```

Environment variables (no secrets are stored in the source tree):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CYBER_THREAT_SECRET_KEY` | random per process | Flask session key |
| `MAX_UPLOAD_MB` | `64` | maximum upload size |
| `MAX_STORED_PREDICTIONS` | `20000` | per-analysis rows stored in SQLite |
| `FLASK_DEBUG` | `0` | set to `1` for the reloader/debugger |
| `PORT` | `5000` | listen port |

---

## Running the Project

```bat
python training/train.py
python app.py
```

Then open <http://127.0.0.1:5000>.

If no dataset is present, generate the bundled sample first:

```bash
python data/make_sample_dataset.py
```

Useful Flask CLI commands:

```bash
flask --app app init-db       # create schema and mirror training metrics
flask --app app reload-model  # reload artifacts without restarting
```

### Live Traffic Sensor (no CSV needed)

The simulation above replays stored records. `utils/live_sensor.py` goes one step further and builds
flow records from **live traffic on an interface you choose**, then scores them with the same model —
no dataset upload required.

```bash
python utils/live_sensor.py --list-interfaces                  # what can be monitored
python utils/live_sensor.py --iface Wi-Fi --seconds 120         # terminal alerts
python utils/live_sensor.py --iface eth0 --csv captures/live.csv
python utils/live_sensor.py --iface eth0 --api http://127.0.0.1:5000/api/sensor/sample
```

How it works:

1. Packets are read on the selected interface and reduced to **flow metadata**: source/destination IP
   and port, protocol, packet and byte counts, forward/reverse counters, duration, mean inter-arrival
   time and TCP flags.
2. `FlowAggregator` builds bidirectional 5-tuple flows and expires a flow after `--flow-timeout`
   seconds of silence (default 20 s). A flow with no reverse traffic is reported as `UNIDIRECTIONAL`.
3. Each completed flow is converted to the canonical training schema, pushed through
   `training/preprocess.py` and scored by the stored pipeline — the model is never retrained.
4. Verdicts are printed in the terminal; with `--csv` they are saved, and with `--api` they are POSTed
   to `/api/sensor/sample`, which stores them in SQLite so they appear in the dashboard, prediction
   history and live simulation as `live-sensor::<interface>`.

```
!!   192.168.1.24 -> 93.184.216.34   TCP  pkts=3120 dir=UNIDIRECTIONAL MALICIOUS  threat= 96.4% DDoS
      192.168.1.24 -> 93.184.216.34   TCP  pkts=40   dir=FORWARD       NORMAL    threat= 32.7%
```

**Prerequisites**

| Platform | Requirement |
| --- | --- |
| Windows | `pip install scapy`, install Npcap (https://npcap.com) with WinPcap compatibility, run the terminal as Administrator |
| Linux / macOS | `pip install scapy`, run as root or grant `CAP_NET_RAW` |

Without them the sensor exits with the exact steps instead of a stack trace:

```
[error] live capture unavailable: Live capture needs the Npcap driver and administrator rights.
  1. Install Npcap from https://npcap.com (keep 'WinPcap API-compatible mode' enabled).
  2. Re-open the terminal with 'Run as administrator'.
```

**Scope and safety** — the sensor is a defensive tool for systems you own or are authorised to
monitor. It reads flow metadata only: payloads are never assembled, parsed, logged or stored, and it
is not a packet interceptor, scanner or offensive tool. Keep `--iface` pointed at your own interfaces;
on shared or untrusted networks, restrict it to loopback (`--iface lo` / the loopback adapter).

`POST /api/sensor/sample` is unauthenticated, exactly like the rest of the web UI, so add
authentication before exposing a deployed instance.

---

## Deploying on Render

The repository ships a Render blueprint, so deployment is a push plus a "Create" click.

1. Push the project to a GitHub repository.
2. In Render choose **New → Blueprint** and select the repository (Render reads `render.yaml`).
3. Render builds with:

   ```bash
   pip install -r requirements.txt
   python data/make_sample_dataset.py     # dataset is not committed to git
   python training/train.py               # trains and writes models/*.pkl at build time
   ```

   and starts with:

   ```bash
   gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 --preload app:app
   ```

4. `CYBER_THREAT_SECRET_KEY` is generated automatically; `/api/health` is used as the health check.

Blueprint highlights:

| Setting | Value | Why |
| --- | --- | --- |
| `--workers 1 --preload` | single preloaded worker | the model bundle is ~25 MB and is shared through `fork` |
| `APP_DATA_DIR=/var/data` | 1 GB disk | SQLite, uploads and reports survive redeploys (Render's filesystem is otherwise ephemeral) |
| `MAX_UPLOAD_MB=32` | smaller cap on the free/starter plan | keeps a single request well inside the 1 GB disk |
| `plan: starter` | change to `free` if you do not need the persistent disk | free instances sleep after inactivity and lose local files |

**Important before going public**

* Add authentication. Every route is currently open, so anyone who finds the URL can upload
  datasets and download reports.
* Free instances spin down and cold-start slowly (loading the model bundle takes a few seconds),
  which suits demos but not monitoring use.
* To use your own data, replace the build command with
  `python training/train.py --dataset data/your_dataset.csv` and commit that CSV, or upload it once
  from the web UI after the first deploy.

Other hosts: the `Procfile` works as-is on Heroku, Railway, Fly.io and any Linux VM
(`gunicorn --bind 0.0.0.0:$PORT app:app`).

---

## Running the tests

```bash
pip install pytest
python -m pytest tests -q
```

30 checks cover the column-mapping layer, the unidirectional direction cascade, cleaning and feature
engineering, the prediction service, every page/API route, the upload → analyse → download flow, path
traversal rejection, the binary and multiclass datasets, and the missing-model error path. Tests that
need trained artifacts skip automatically when `models/metadata.json` is absent.

---

## API Documentation

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Home page |
| GET | `/dashboard` | Cybersecurity dashboard |
| GET | `/upload` | Upload form and dataset preview |
| POST | `/predict` | Analyse a dataset (multipart form, field `dataset` or `stored_filename`) |
| GET | `/results/<id>` | Detection results for one analysis |
| GET | `/metrics` | Model performance, confusion matrices, ROC, importance |
| GET | `/download/<filename>` | Download a prediction CSV |
| GET | `/live` | Live traffic simulation |
| GET | `/about` | Project documentation |
| POST | `/api/predict` | JSON prediction endpoint |
| GET | `/api/simulation/<id>` | Random stored records (live simulation feed) |
| GET | `/api/metrics` | Training metrics as JSON |
| GET | `/api/health` | Service and model status |

### `POST /api/predict`

```bash
curl -X POST -F "dataset=@data/traffic_dataset.csv" http://127.0.0.1:5000/api/predict
```

```json
{
  "total_records": 1000,
  "normal": 720,
  "malicious": 280,
  "threat_percentage": 28.0,
  "unidirectional_records": 143,
  "model_name": "Gradient Boosting",
  "threshold": 0.5,
  "attack_types": {"Port Scan": 90, "DDoS": 60},
  "protocol_distribution": {"TCP": 810, "UDP": 150, "ICMP": 40},
  "direction_distribution": {"FORWARD": 520, "REVERSE": 337, "UNIDIRECTIONAL": 143},
  "warnings": [],
  "predictions_sample": [
    {"Source IP": "192.168.10.4", "Prediction": "MALICIOUS", "Threat Probability": 94.7}
  ]
}
```

Errors return the same shape with an `error` key, e.g.
`{"error": "Invalid file type. Only .csv datasets are accepted", "status": 400}`.

---

## Project Structure

```
cyber-threat-detection/
├── app.py                     # Flask app, routes, SQLAlchemy models, security, API
├── requirements.txt
├── README.md
├── data/
│   ├── traffic_dataset.csv    # bundled sample dataset
│   └── make_sample_dataset.py # reproducible generator
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl
│   ├── attack_classifier.pkl
│   └── scaler.pkl             # fitted preprocessing pipeline + metadata.json
├── training/
│   ├── train.py               # training, comparison, artifact persistence
│   ├── preprocess.py          # mapping, cleaning, direction logic, features
│   └── evaluate.py            # metrics, confusion matrices, ROC, plots
├── utils/
│   └── prediction.py          # inference service (no retraining)
├── templates/
│   ├── base.html  index.html  dashboard.html  upload.html
│   ├── results.html  metrics.html  live.html  about.html  error.html
│   └── _macros.html
├── static/
│   ├── css/style.css
│   ├── js/dashboard.js        # Chart.js dashboards + simulation engine
│   └── images/generated/      # training figures
├── database/
│   └── cyber_threat.db        # uploads · predictions · model_metrics
├── tests/
│   └── test_pipeline.py       # 30 automated checks (pytest)
├── uploads/                   # sanitised uploaded files
└── reports/                   # downloadable prediction CSVs
```

---

## Screenshots

<!-- Replace the placeholders below with real captures of your run -->

| Screen | Placeholder |
| --- | --- |
| Home page | `docs/screenshots/home.png` |
| Upload and preview | `docs/screenshots/upload.png` |
| Dashboard | `docs/screenshots/dashboard.png` |
| Detection results | `docs/screenshots/results.png` |
| Model performance | `docs/screenshots/metrics.png` |
| Live simulation | `docs/screenshots/live.png` |

---

## Security

* Filenames are sanitised with `werkzeug.secure_filename` and prefixed by a timestamp; the original
  name is never used as a path.
* Only `.csv` is accepted; other extensions are rejected with a clear message.
* Uploads are capped by `MAX_CONTENT_LENGTH`; exceeding it returns a 413 page.
* The Flask secret key comes from `CYBER_THREAT_SECRET_KEY`; a random key is generated when it is
  absent, and no credential is hard-coded.
* Database access uses SQLAlchemy ORM queries only — no string-built SQL.
* Uploaded content is parsed strictly as data and never executed.
* All failures (missing dataset, invalid CSV, empty CSV, missing label, missing model, unknown
  columns, size limit, 404/500) render a friendly page instead of a traceback.

---

## Future Improvements

* Unsupervised anomaly detection (autoencoders, isolation forest) for unknown attack families.
* Online/streaming learning with concept-drift monitoring.
* Per-record SHAP explanations in the results table.
* Time-series features (periodicity, beaconing autocorrelation) for C2 detection.
* Role-based access, per-user report history and PDF export.
* Containerisation (Docker) and a documented deployment path.

---

## Limitations

* Flow-level features cannot attribute an attack to a user or a host process.
* Encrypted payloads are opaque; only metadata is analysed.
* The bundled dataset is synthetic; absolute accuracy does not transfer to a production network.
* Models drift as traffic mix changes and must be retrained periodically.
* The live simulation replays stored records — it is not live packet interception.

---

## Conclusion

The project delivers a working, end-to-end threat-detection system: tolerant data ingestion,
reproducible and leakage-free training, comparison of five classifiers with a full metric bundle,
consistent training/prediction preprocessing, a dark cybersecurity dashboard, a live simulation,
SQLite persistence, CSV reporting and a documented JSON API. Every displayed metric is computed
from real data at run time, and every failure path produces an actionable message rather than a
stack trace — the behaviour expected of a deployable academic project rather than a notebook
prototype.