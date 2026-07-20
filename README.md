# transaction-fraud-detector

An end-to-end MLOps project for detecting fraudulent credit card transactions. Built in 10 stages, covering the full lifecycle from raw data to a served model with monitoring, automated retraining, and deployment gating.

---

## What this project covers

- Data validation and preprocessing pipeline
- XGBoost model training with experiment tracking (MLflow)
- Automated model promotion gating (threshold checks + champion/challenger comparison)
- REST API serving with FastAPI + pydantic input validation
- CI/CD with GitHub Actions (tests + model validation gate on every push)
- Shadow mode, canary deployment simulation, and A/B testing with statistical significance
- Data drift detection (Evidently + Wasserstein distance) and rolling performance monitoring
- Automated retraining pipeline triggered by drift or performance degradation

---

## Architecture

```mermaid
flowchart TD
    A[Raw Data\ncreditcard.csv] --> B[Validation\nsrc/validate.py]
    B --> C[Preprocessing\nsrc/preprocess.py]
    C --> D[Train Models\nsrc/train.py]
    D --> E[MLflow\nExperiment Tracking]
    E --> F[Promotion Gate\nsrc/promote.py]
    F -->|passes| G[Production Model\nmodels/production_model.joblib]
    F -->|fails| X[Rejected\nchampion unchanged]
    G --> H[FastAPI\nsrc/app.py]
    H --> I[/predict endpoint\nfraud probability + label]

    G --> J[Shadow Mode\nscripts/shadow_mode.py]
    G --> K[Canary\nscripts/canary.py]
    K --> L[A/B Analysis\nscripts/ab_analysis.py]

    G --> M[Drift Monitor\nscripts/drift_report.py]
    G --> N[Performance Monitor\nscripts/performance_monitor.py]
    M --> O[Retraining Pipeline\nscripts/retrain_pipeline.py]
    N --> O
    O --> E
```

**The flow in plain text:**

1. Raw CSV → validated → preprocessed → train/test split saved
2. XGBoost and logistic regression trained, both logged to MLflow
3. Promotion gate checks thresholds and compares against current champion
4. Approved model served via FastAPI at `/predict`
5. Shadow mode runs challenger silently alongside champion
6. Canary routes 5% of traffic to challenger, A/B analysis checks significance
7. Evidently drift report + rolling performance monitor watch for degradation
8. If drift or degradation detected → retrain pipeline fires → new challenger logged to MLflow

---

## How to run

### Setup

```bash
pip install -r requirements.txt
```

### Data

Place `creditcard.csv` in `data/raw/`. Dataset available at:
https://www.kaggle.com/mlg-ulb/creditcardfraud

### Run each stage

```bash
# explore the data
python notebooks/01_exploration.py

# validate and preprocess
python -c "
import pandas as pd, sys
sys.path.insert(0, 'src')
from validate import validate
from preprocess import preprocess
df = pd.read_csv('data/raw/creditcard.csv')
validate(df)
preprocess(df)
"

# train models and log to mlflow
python src/train.py

# view mlflow ui (run in a separate terminal)
mlflow ui

# run promotion gate
python src/promote.py

# run tests
python -m pytest tests/ -v

# serve the api
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

# shadow mode
python scripts/shadow_mode.py

# canary simulation
python scripts/canary.py

# a/b analysis
python scripts/ab_analysis.py

# drift report (saves to reports/drift_report.html)
python scripts/drift_report.py

# performance monitor
python scripts/performance_monitor.py

# full retraining pipeline
python scripts/retrain_pipeline.py

# model threshold gate (used in ci)
python scripts/check_model_thresholds.py
```

### Run with Docker

```bash
# build and run the api
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

API endpoints:
- `GET /health` — service health check
- `POST /predict` — send transaction features, get fraud probability and label
- `GET /docs` — interactive API documentation (FastAPI built-in)

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406.0, "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99,
       "V5": -0.52, "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77,
       "V10": -2.77, "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.29,
       "V15": 0.39, "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42,
       "V20": 0.13, "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32,
       "V25": 0.04, "V26": 0.18, "V27": 0.26, "V28": -0.14, "Amount": 0.0}'
```

---

## Tools and why

| Tool | Used for | Why this over alternatives |
|------|----------|---------------------------|
| XGBoost | Primary model | Native scale_pos_weight for imbalance, strong on tabular data, well understood |
| MLflow | Experiment tracking | Logs params/metrics/artifacts, run comparison UI, standard in MLOps interviews |
| FastAPI | Model serving | Built-in pydantic validation, auto docs, async-ready, industry standard for ML APIs |
| Pydantic | Input validation | Declarative, field-level error messages, zero manual validation code |
| Evidently | Drift detection | Open source, HTML reports, Wasserstein distance for numerical drift |
| pytest | Testing | Standard Python testing, clean fixtures, works with CI |
| GitHub Actions | CI/CD | No setup needed for GitHub repos, simple YAML, runs on every push |
| scikit-learn | Preprocessing, LR | ColumnTransformer/Pipeline for consistent train/inference transforms |
| scipy | Stats tests | Fisher's exact test for A/B significance on small fraud counts |
| joblib | Model serialization | Standard for sklearn objects, faster than pickle for numpy arrays |

---

## Project structure

```
transaction-fraud-detector/
├── data/
│   ├── raw/              # creditcard.csv (gitignored)
│   └── processed/        # train/test splits (gitignored)
├── models/               # saved model files (gitignored)
├── notebooks/
│   └── 01_exploration.py
├── reports/              # evidently html reports
├── scripts/
│   ├── shadow_mode.py
│   ├── canary.py
│   ├── ab_analysis.py
│   ├── drift_report.py
│   ├── performance_monitor.py
│   ├── retrain_pipeline.py
│   └── check_model_thresholds.py
├── src/
│   ├── validate.py
│   ├── preprocess.py
│   ├── train.py
│   ├── promote.py
│   └── app.py
├── tests/
│   ├── test_validate.py
│   └── test_promote.py
├── docs/                 # phase docs (gitignored, local only)
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
