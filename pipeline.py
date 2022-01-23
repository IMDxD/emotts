import argparse

from src.trainer_feature import Trainer
from src.inferencer import Inferencer
from src.trainer_hifi import HIFITrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    trainer_feature = Trainer(args.config)
    inferencer_feature = Inferencer(args.config)
    trainer_hifi = HIFITrainer(args.config)
    trainer_feature.train()
    inferencer_feature.proceed_data()
    trainer_hifi.train()


if __name__ == "__main__":
    main()
