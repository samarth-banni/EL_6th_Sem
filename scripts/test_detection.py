from __future__ import annotations

import argparse
from pathlib import Path

from core.detector import TrafficDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO12 traffic detection against a local image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/yolo12n.pt"))
    parser.add_argument("--road-capacity", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = TrafficDetector(model_path=args.model, road_capacity=args.road_capacity)
    result = detector.analyze_bytes(args.image.read_bytes())
    print(result)


if __name__ == "__main__":
    main()
