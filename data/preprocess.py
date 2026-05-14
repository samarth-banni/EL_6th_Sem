from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_dataset(destination: Path = RAW_DIR / "Metro_Interstate_Traffic_Volume.csv") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    gz_path = destination.with_suffix(destination.suffix + ".gz")

    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    gz_path.write_bytes(response.content)

    with gzip.open(gz_path, "rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    return destination


def preprocess(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    frame["date_time"] = pd.to_datetime(frame["date_time"])
    frame = frame.sort_values("date_time").reset_index(drop=True)

    frame["hour"] = frame["date_time"].dt.hour
    frame["day_of_week"] = frame["date_time"].dt.dayofweek
    frame["month"] = frame["date_time"].dt.month
    frame["station_id"] = "1"
    frame["time_idx"] = range(len(frame))

    for column in ["holiday", "weather_main"]:
        encoder = LabelEncoder()
        frame[f"{column}_encoded"] = encoder.fit_transform(frame[column].fillna("None"))

    continuous_columns = ["temp", "rain_1h", "snow_1h"]
    scaler = StandardScaler()
    frame[continuous_columns] = scaler.fit_transform(frame[continuous_columns].fillna(0.0))

    selected_columns = [
        "time_idx",
        "station_id",
        "traffic_volume",
        "hour",
        "day_of_week",
        "month",
        "holiday_encoded",
        "weather_main_encoded",
        "temp",
        "rain_1h",
        "snow_1h",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame[selected_columns].to_csv(output_csv, index=False)
    return frame[selected_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Metro Interstate Traffic Volume data for TFT.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "Metro_Interstate_Traffic_Volume.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "metro_traffic_processed.csv")
    parser.add_argument("--download", action="store_true", help="Download the raw dataset before preprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = download_dataset(args.input) if args.download or not args.input.exists() else args.input
    preprocess(input_csv, args.output)
    print(f"Processed TFT dataset written to {args.output}")


if __name__ == "__main__":
    main()
