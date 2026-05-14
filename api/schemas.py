from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    detector_loaded: bool
    forecaster_loaded: bool
    timestamp: datetime


class FrameRequest(BaseModel):
    frame_base64: str = Field(..., description="Base64 encoded image frame.")
    road_capacity: float = Field(100.0, gt=0, description="Estimated vehicle capacity for the road segment.")


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    area_ratio: float


class FrameAnalysisResponse(BaseModel):
    total_vehicles: int
    counts: dict[str, int]
    chaos_index: float
    average_area_occupied: float
    boxes: list[BoundingBox]
    timestamp: datetime


class VideoFrameResult(BaseModel):
    frame_index: int
    timestamp_seconds: float
    analysis: FrameAnalysisResponse


class VideoAnalysisResponse(BaseModel):
    filename: str
    sampled_frames: int
    average_vehicle_count: float
    max_vehicle_count: int
    average_chaos_index: float
    frames: list[VideoFrameResult]


class TelegramStatusResponse(BaseModel):
    configured: bool


class AlertRequest(BaseModel):
    source: str = "dashboard"
    vehicle_count: float = 0.0
    chaos_index: float = 0.0
    predicted_volume_30min: float | None = None
    prediction_level: str | None = None
    tft_confidence: float | None = None
    vehicle_density: str | None = None
    location: str | None = None
    weather: str | None = None
    time_period: str | None = None
    reason: str | None = None
    message: str | None = None


class AlertResponse(BaseModel):
    sent: bool
    detail: str


class RiskRequest(BaseModel):
    source: str = "live dashboard"
    vehicle_count: float = 0.0
    chaos_index: float = 0.0
    weather: str = "sunny"
    location: str = "unknown"
    chaos_threshold: float = 0.002
    volume_threshold: float = 4500.0
    send_alert: bool = False


class RiskResponse(BaseModel):
    risk_level: str
    should_alert: bool
    alert_sent: bool
    alert_detail: str | None = None
    time_period: str
    weather: str
    location: str
    reason: str
    prediction_level: str
    tft_confidence: float
    vehicle_density: str
    predicted_volume_30min: float
    forecast: ForecastResponse


class ForecastPoint(BaseModel):
    horizon_minutes: int
    predicted_volume: float


class ForecastResponse(BaseModel):
    generated_at: datetime
    prediction_window_minutes: int = 30
    interval_minutes: int = 5
    predictions: list[ForecastPoint]
    interpretation: dict[str, float] = Field(
        default_factory=dict,
        description="Normalized feature/covariate importance scores returned by the TFT explainer.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
