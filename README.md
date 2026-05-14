# 6th Sem EL - Traffic Intelligence Engine

Real-time traffic monitoring and 30-minute traffic-volume forecasting backend built with FastAPI, Ultralytics YOLO12, and PyTorch Forecasting Temporal Fusion Transformer (TFT).

## Features

- YOLO12 spatial analyzer for cars, motorcycles, buses, and trucks.
- Chaos Index: `(total_vehicles * average_area_occupied) / road_capacity`.
- Five-minute aggregation buffer for detector outputs.
- Metro Interstate Traffic Volume preprocessing pipeline.
- TFT training scaffold using `TimeSeriesDataSet`.
- FastAPI endpoints with Pydantic request and response models.
- Docker and Render deployment configuration.

## Project Structure

```text
api/        FastAPI app and Pydantic schemas
core/       YOLO detector and TFT forecasting service
data/       Dataset download, preprocessing, and training scripts
models/     Local model artifacts (ignored by git)
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Place model artifacts in `models/`:

- `models/yolo12n.pt`
- `models/tft_weights.ckpt`

If `yolo12n.pt` is missing, Ultralytics will attempt to resolve the model name when the detector loads.

## Data Pipeline

```bash
python data/preprocess.py --download --output data/processed/metro_traffic_processed.csv
python data/train_tft.py --data data/processed/metro_traffic_processed.csv --max-epochs 5
```

The preprocessing script downloads the Metro Interstate Traffic Volume CSV, creates time features, label-encodes categorical weather fields, standardizes continuous weather values, and emits a TFT-ready CSV.

## API

Start the service:

```bash
uvicorn api.main:app --reload
```

Frontend dashboard:

```text
http://127.0.0.1:8000/app/index.html
```

Endpoints:

- `GET /health`
- `POST /stream/frame`
- `POST /stream/video`
- `GET /forecast/30min`
- `GET /alerts/telegram/status`
- `POST /alerts/telegram`

## Telegram Alerts

Create a Telegram bot with BotFather, get your bot token, then get your chat ID from `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` after sending one message to the bot.

PowerShell setup:

```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
uvicorn api.main:app --reload
```

Example frame request:

```json
{
  "frame_base64": "<base64 encoded image>",
  "road_capacity": 120
}
```

## Docker

```bash
docker build -t traffic-intelligence-engine .
docker run -p 8000:8000 traffic-intelligence-engine
```
