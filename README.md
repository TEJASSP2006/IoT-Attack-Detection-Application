# IoT Attack Detection App (CICIoT2023)

End-to-end application for:

1. Training on CICIoT2023
2. Real-time inference (API + optional MQTT)
3. Prediction logging to SQLite
4. Attack alerting (Telegram and/or email)
5. API key protection
6. Docker deployment

## What is now automatic

- Loading and concatenating all CSV files from `archive/wataiData/csv/CICIoT2023` during training
- Chunked dataset streaming to avoid reading all files into memory at once
- Automatic row capping (`--max-rows`) with random sampling across all CSV files when dataset is too large
- Label column detection (`--label-column auto` by default)
- Preprocessing for mixed data types:
  - Numeric: median impute + scale
  - Categorical: most-frequent impute + one-hot encode
- Class imbalance support (`class_weight=balanced_subsample`)
- Attack threshold tuning on validation split
- Prediction logging in `data/predictions.db`
- Optional alerts on attack predictions

## Project Structure

- `train.py` - training pipeline + auto label detection + threshold tuning
- `realtime_detector.py` - inference engine + MQTT support + DB logging + alerts
- `api.py` - FastAPI app (dashboard + REST API + auth)
- `alerts.py` - Telegram/email alert dispatch
- `storage.py` - SQLite persistence for predictions
- `models/` - saved model and metadata
- `archive/wataiData/` - your dataset location

## 1) Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**PowerShell: “running scripts is disabled”** when activating the venv:

- **Fix policy (current user):** `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Or bypass this session only:** `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` then run `Activate.ps1` again
- **Or skip activation** and use the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## 2) Train model (automatic label detection)

Default dataset path is already `archive/wataiData/csv/CICIoT2023`.

```powershell
python train.py
```

Default memory-safe options:
- `--max-rows 1500000`
- `--chunksize 200000`

To use more rows (if your RAM allows):

```powershell
python train.py --data-path "archive\wataiData\csv\CICIoT2023" --max-rows 3000000 --chunksize 250000
```

To disable sampling and try full-data training (high RAM required):

```powershell
python train.py --data-path "archive\wataiData\csv\CICIoT2023" --max-rows 0
```

If you want explicit control:

```powershell
python train.py --data-path "archive\wataiData\csv\CICIoT2023" --label-column auto
```

If you already know the exact label column:

```powershell
python train.py --label-column label
```

Artifacts generated:
- `models\iot_model.joblib`
- `models\model_metadata.json`

## 3) Secure API key (recommended)

Copy `.env.example` to `.env` and set `API_KEY`:

```powershell
copy .env.example .env
```

If `API_KEY` is set, send header on protected endpoints:
- `x-api-key: <your-key>`

Protected endpoints:
- `POST /predict`
- `GET /predictions`

## 4) Run API

```powershell
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Windows note:** If `uvicorn` alone fails with “not recognized”, always use `python -m uvicorn` (and ensure the venv is activated and `pip install -r requirements.txt` completed).

**Browser note:** `--host 0.0.0.0` only means “listen on all interfaces.” Do **not** open `http://0.0.0.0:8000` in Chrome — it often times out. Use **`http://127.0.0.1:8000`** or **`http://localhost:8000`** instead.

Open:
- **Web dashboard** (metrics + Single / Batch / **Live Monitor**): `http://localhost:8000/`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

The dashboard is served by this same FastAPI app (port **8000**). If you use another frontend on port 5000, point it at `http://127.0.0.1:8000` for API calls, or open the built-in dashboard URL above.

**Dashboard:** If `API_KEY` is set, open the **☰ menu** (top-right) → **Settings** → paste key → **Save key**.

**API additions:**
- `GET /stats` — totals, attacks, benign, detection rate (from SQLite)
- `POST /predict/batch` — body `{ "records": [ {...}, ... ] }` (max 100)
- `POST /predict/batch_csv` — multipart form field `file` (`.csv`), up to 500 rows; common label columns are dropped automatically

## 5) Test prediction

```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -H "x-api-key: change-me" `
  -d "{\"record\": {\"flow_duration\": 120, \"pkt_len_mean\": 64.7}}"
```

## 6) Check stored predictions

```powershell
curl -X GET "http://localhost:8000/predictions?limit=20" `
  -H "x-api-key: change-me"
```

Data is stored in:
- `data/predictions.db`

## 7) Optional MQTT real-time stream

```powershell
python realtime_detector.py --mqtt-host localhost --mqtt-port 1883 --mqtt-topic iot/flow
```

MQTT payload must be JSON object with feature keys.

## 8) Optional attack alerts

Configure `.env`:

- Telegram:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
- Email:
  - `ALERT_SMTP_HOST`
  - `ALERT_SMTP_PORT`
  - `ALERT_SMTP_USER`
  - `ALERT_SMTP_PASSWORD`
  - `ALERT_EMAIL_FROM`
  - `ALERT_EMAIL_TO`

When `is_attack=true`, alerts are dispatched if config exists.

## 9) Docker deployment

Build and run:

```powershell
docker compose up --build
```

The service runs on port `8000`.
