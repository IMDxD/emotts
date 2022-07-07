import argparse

from src.trainer_feature_adversarial import Trainer as TrainerAdv
from src.trainer_feature_reversal import Trainer as TrainerRes
from src.train_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config.loss.is_reversal:
        trainer = TrainerRes(config)
    else:
        trainer = TrainerAdv(config)
    trainer.train()


if __name__ == "__main__":
    main()
