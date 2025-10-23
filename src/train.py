"""CLI entrypoint for training models from YAML configuration files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import ExperimentConfig, load_config
from training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEFT LoRA-GA training runner.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config)
    logging.getLogger(__name__).info("Loaded configuration from %s", args.config)
    metrics = run_training(config)
    logging.info("Training finished. Metrics: %s", metrics)


if __name__ == "__main__":
    main()

