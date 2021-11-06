import argparse

from src import Inferencer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='configuration file path'
    )
    args = parser.parse_args()
    inferencer = Inferencer(args.config)
    inferencer.feature_model.to("cuda")
    inferencer.proceed_data()


if __name__ == '__main__':
    main()
