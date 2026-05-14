from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 6


def build_dataset(frame: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    training_cutoff = int(frame["time_idx"].max()) - MAX_PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        frame[lambda item: item.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="traffic_volume",
        group_ids=["station_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["station_id"],
        time_varying_known_reals=["time_idx", "hour", "day_of_week", "month"],
        time_varying_unknown_reals=["traffic_volume", "temp", "rain_1h", "snow_1h"],
        time_varying_known_categoricals=["holiday_encoded", "weather_main_encoded"],
        target_normalizer=GroupNormalizer(groups=["station_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        frame,
        predict=True,
        stop_randomization=True,
    )
    return training, validation


def train(data_path: Path, max_epochs: int, batch_size: int, output_dir: Path) -> Path:
    frame = pd.read_csv(data_path)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["holiday_encoded"] = frame["holiday_encoded"].astype(str)
    frame["weather_main_encoded"] = frame["weather_main_encoded"].astype(str)

    training, validation = build_dataset(frame)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="adam",
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="tft_weights",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[checkpoint],
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return Path(checkpoint.best_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TFT model for 30-minute traffic-volume forecasting.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/metro_traffic_processed.csv"))
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = train(args.data, args.max_epochs, args.batch_size, args.output_dir)
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
