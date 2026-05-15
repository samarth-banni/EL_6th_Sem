from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_IMPORTANCE = {
    "hour": 0.24,
    "day_of_week": 0.18,
    "month": 0.08,
    "temp": 0.12,
    "rain_1h": 0.16,
    "snow_1h": 0.06,
    "holiday_encoded": 0.07,
    "weather_main_encoded": 0.09,
}


class TrafficForecaster:
    """TFT inference wrapper with explainability output."""

    def __init__(self, checkpoint_path: str | Path = "models/tft_weights.ckpt") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model: object | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is None and self.checkpoint_path.exists():
            from pytorch_forecasting import TemporalFusionTransformer

            self.model = TemporalFusionTransformer.load_from_checkpoint(str(self.checkpoint_path))
            self.model.eval()

    def predict_next_30_minutes(self, recent_observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.load()
        if self.model is None:
            return self._fallback_prediction(recent_observations or [])

        frame = self._build_inference_frame(recent_observations or [])
        import torch

        with torch.no_grad():
            raw_prediction = self.model.predict(frame, mode="raw", return_x=True)

        prediction_values = raw_prediction.output.prediction.detach().cpu().numpy().reshape(-1)[:6]
        interpretation = self._interpret(raw_prediction)
        return self._format_response(prediction_values, interpretation, source="tft")

    def _fallback_prediction(self, recent_observations: list[dict[str, Any]]) -> dict[str, Any]:
        baseline = 30.0
        if recent_observations:
            vehicle_counts = [float(item.get("total_vehicles", 0.0)) for item in recent_observations]
            chaos = [float(item.get("chaos_index", 0.0)) for item in recent_observations]
            baseline = max(float(np.mean(vehicle_counts)) * 12.0, 1.0)
            baseline *= 1.0 + min(float(np.mean(chaos)), 1.0)

        trend = np.linspace(1.0, 1.12, 6)
        prediction_values = baseline * trend
        return self._format_response(prediction_values, DEFAULT_IMPORTANCE, source="heuristic_fallback")

    @staticmethod
    def _build_inference_frame(recent_observations: list[dict[str, Any]]) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        rows = []
        total_rows = 30
        for idx in range(total_rows):
            source = recent_observations[idx % len(recent_observations)] if recent_observations and idx < 24 else {}
            rows.append(
                {
                    "time_idx": idx,
                    "station_id": "1",
                    "traffic_volume": float(source.get("total_vehicles", 0.0)),
                    "hour": now.hour,
                    "day_of_week": now.weekday(),
                    "month": now.month,
                    "temp": 0.0,
                    "rain_1h": 0.0,
                    "snow_1h": 0.0,
                    "holiday_encoded": "0",
                    "weather_main_encoded": "0",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _interpret(raw_prediction: Any) -> dict[str, float]:
        try:
            interpretation = raw_prediction.output.interpretation
            importance = interpretation.get("encoder_variables", {})
            total = sum(float(value) for value in importance.values()) or 1.0
            return {str(key): float(value) / total for key, value in importance.items()}
        except Exception:
            return DEFAULT_IMPORTANCE

    @staticmethod
    def _format_response(prediction_values: np.ndarray, interpretation: dict[str, float], source: str) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc),
            "predictions": [
                {"horizon_minutes": (index + 1) * 5, "predicted_volume": float(value)}
                for index, value in enumerate(prediction_values[:6])
            ],
            "interpretation": interpretation,
            "metadata": {"model_source": source},
        }
