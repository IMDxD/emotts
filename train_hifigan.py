import argparse

from src.trainer_hifi import HIFITrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    trainer = HIFITrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
