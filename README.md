# transaction-fraud-detector

End-to-end MLOps project for detecting fraudulent credit card transactions.

---

## How to run locally

```bash
pip install -r requirements.txt
pip install fastapi uvicorn

python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

- `GET /health` — check if the service is up
- `POST /predict` — send a transaction, get back fraud probability and label
- `GET /docs` — auto-generated interactive API docs (FastAPI built-in)

## How to run with Docker

```bash
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406.0, "V1": -2.31, "V2": 1.95, ...all 30 features..., "Amount": 0.0}'
```

Response:
```json
{
  "fraud_probability": 0.999,
  "label": "fraud",
  "threshold_used": 0.5
}
```
