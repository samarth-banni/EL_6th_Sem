from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import ValidationError

from api.schemas import (
    AlertRequest,
    AlertResponse,
    ForecastResponse,
    FrameAnalysisResponse,
    FrameRequest,
    HealthResponse,
    RiskRequest,
    RiskResponse,
    TelegramStatusResponse,
    VideoAnalysisResponse,
)
from core.alerts import TelegramAlerter, build_traffic_alert_message
from core.detector import TrafficDetector
from core.forecasting import TrafficForecaster


load_dotenv()

app = FastAPI(
    title="Traffic Intelligence Engine",
    version="0.1.0",
    description="Real-time YOLO12 traffic analyzer with 30-minute TFT forecasting.",
)

detector = TrafficDetector()
forecaster = TrafficForecaster()
alerter = TelegramAlerter()

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.on_event("startup")
async def startup() -> None:
    try:
        detector.load()
    except Exception:
        # The service should still boot in environments where weights are mounted later.
        pass

    try:
        forecaster.load()
    except Exception:
        pass


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        detector_loaded=detector.is_loaded,
        forecaster_loaded=forecaster.is_loaded,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/app/index.html")


@app.post("/stream/frame", response_model=FrameAnalysisResponse)
async def stream_frame(request: Request) -> FrameAnalysisResponse:
    try:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=422, detail="Multipart requests must include a file field.")
            frame_bytes = await upload.read()
            analysis = detector.analyze_bytes(frame_bytes)
        elif content_type.startswith("application/octet-stream") or content_type.startswith("image/"):
            frame_bytes = await request.body()
            analysis = detector.analyze_bytes(frame_bytes)
        else:
            payload = FrameRequest.model_validate(await request.json())
            analysis = detector.analyze_base64(payload.frame_base64, road_capacity=payload.road_capacity)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=503, detail=f"Detector unavailable: {exc}") from exc

    return FrameAnalysisResponse(**analysis)


@app.post("/stream/video", response_model=VideoAnalysisResponse)
async def stream_video(
    file: UploadFile = File(...),
    sample_every_seconds: float = 2.0,
    road_capacity: float = 100.0,
    max_frames: int = 60,
) -> VideoAnalysisResponse:
    suffix = Path(file.filename or "traffic.mp4").suffix or ".mp4"
    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        analysis = detector.analyze_video(
            temp_path,
            sample_every_seconds=sample_every_seconds,
            road_capacity=road_capacity,
            max_frames=max_frames,
        )
        return VideoAnalysisResponse(filename=file.filename or "uploaded-video", **analysis)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Video analysis unavailable: {exc}") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.get("/forecast/30min", response_model=ForecastResponse)
async def forecast_30min() -> ForecastResponse:
    recent_observations = detector.get_recent_aggregates()
    forecast = forecaster.predict_next_30_minutes(recent_observations)
    return ForecastResponse(**forecast)


@app.get("/alerts/telegram/status", response_model=TelegramStatusResponse)
async def telegram_status() -> TelegramStatusResponse:
    return TelegramStatusResponse(configured=alerter.is_configured)


@app.post("/alerts/telegram", response_model=AlertResponse)
async def send_telegram_alert(payload: AlertRequest) -> AlertResponse:
    message = payload.message or build_traffic_alert_message(
        source=payload.source,
        vehicle_count=payload.vehicle_count,
        chaos_index=payload.chaos_index,
        predicted_volume_30min=payload.predicted_volume_30min,
        prediction_level=payload.prediction_level,
        tft_confidence=payload.tft_confidence,
        vehicle_density=payload.vehicle_density,
        location=payload.location,
        weather=payload.weather,
        time_period=payload.time_period,
        reason=payload.reason,
    )
    result = alerter.send_message(message)
    return AlertResponse(sent=result.sent, detail=result.detail)


@app.post("/risk/evaluate", response_model=RiskResponse)
async def evaluate_risk(payload: RiskRequest) -> RiskResponse:
    forecast = ForecastResponse(**forecaster.predict_next_30_minutes(detector.get_recent_aggregates()))
    predicted_30 = forecast.predictions[-1].predicted_volume if forecast.predictions else 0.0
    time_period = current_time_period()
    prediction_level = prediction_level_for(predicted_30, payload.volume_threshold)
    vehicle_density = vehicle_density_for(payload.vehicle_count)
    tft_confidence = estimate_tft_confidence(forecast, payload.vehicle_count)
    risk_level, should_alert, reason = explain_risk(
        vehicle_count=payload.vehicle_count,
        chaos_index=payload.chaos_index,
        predicted_volume_30min=predicted_30,
        prediction_level=prediction_level,
        vehicle_density=vehicle_density,
        weather=payload.weather,
        time_period=time_period,
        chaos_threshold=payload.chaos_threshold,
        volume_threshold=payload.volume_threshold,
    )

    alert_sent = False
    alert_detail = None
    if payload.send_alert and should_alert:
        result = alerter.send_message(
            build_traffic_alert_message(
                source=payload.source,
                vehicle_count=payload.vehicle_count,
                chaos_index=payload.chaos_index,
                predicted_volume_30min=predicted_30,
                prediction_level=prediction_level,
                tft_confidence=tft_confidence,
                vehicle_density=vehicle_density,
                location=payload.location,
                weather=payload.weather,
                time_period=time_period,
                reason=reason,
            )
        )
        alert_sent = result.sent
        alert_detail = result.detail

    return RiskResponse(
        risk_level=risk_level,
        should_alert=should_alert,
        alert_sent=alert_sent,
        alert_detail=alert_detail,
        time_period=time_period,
        weather=payload.weather,
        location=payload.location,
        reason=reason,
        prediction_level=prediction_level,
        tft_confidence=tft_confidence,
        vehicle_density=vehicle_density,
        predicted_volume_30min=predicted_30,
        forecast=forecast,
    )


def current_time_period() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 16:
        return "afternoon"
    if 16 <= hour < 21:
        return "evening"
    return "night"


def explain_risk(
    vehicle_count: float,
    chaos_index: float,
    predicted_volume_30min: float,
    prediction_level: str,
    vehicle_density: str,
    weather: str,
    time_period: str,
    chaos_threshold: float,
    volume_threshold: float,
) -> tuple[str, bool, str]:
    weather_factor = {
        "rain": 1.25,
        "fog": 1.2,
        "cloudy": 1.08,
        "sunny": 1.0,
    }.get(weather.lower(), 1.0)
    time_factor = {
        "morning": 1.15,
        "evening": 1.2,
        "afternoon": 1.0,
        "night": 0.85,
    }.get(time_period, 1.0)

    adjusted_volume = predicted_volume_30min * weather_factor * time_factor
    high_chaos = chaos_index >= chaos_threshold
    high_forecast = adjusted_volume >= volume_threshold
    heavy_count = vehicle_density == "high"

    reasons = []
    if high_chaos:
        reasons.append("live chaos index crossed the congestion threshold")
    if high_forecast:
        reasons.append(f"30-minute TFT prediction is {prediction_level} after time and weather adjustment")
    if heavy_count:
        reasons.append("current vehicle count is heavy")
    if weather.lower() in {"rain", "fog"}:
        reasons.append(f"{weather.lower()} weather can slow traffic flow")
    if time_period in {"morning", "evening"}:
        reasons.append(f"{time_period} peak period increases congestion risk")

    score = int(high_chaos) + int(high_forecast) + int(heavy_count)
    if score >= 2:
        level = "high"
    elif score == 1:
        level = "medium"
    else:
        level = "low"

    reason = "; ".join(reasons) if reasons else "traffic is currently within normal operating range"
    return level, level == "high", reason


def prediction_level_for(predicted_volume_30min: float, high_threshold: float) -> str:
    medium_threshold = high_threshold * 0.55
    if predicted_volume_30min >= high_threshold:
        return "high"
    if predicted_volume_30min >= medium_threshold:
        return "medium"
    return "low"


def vehicle_density_for(vehicle_count: float) -> str:
    if vehicle_count >= 20:
        return "high"
    if vehicle_count >= 8:
        return "medium"
    return "low"


def estimate_tft_confidence(forecast: ForecastResponse, vehicle_count: float) -> float:
    source = forecast.metadata.get("model_source")
    base = 86.0 if source == "tft" else 62.0
    values = [point.predicted_volume for point in forecast.predictions]
    if len(values) >= 2:
        mean_value = sum(values) / len(values)
        if mean_value:
            volatility = (max(values) - min(values)) / mean_value
            base -= min(volatility * 8.0, 10.0)
    if vehicle_count <= 0:
        base -= 8.0
    elif vehicle_count >= 8:
        base += 3.0
    return max(50.0, min(base, 95.0))
