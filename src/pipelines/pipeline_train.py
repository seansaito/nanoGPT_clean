import argparse
import logging

from src.utils import set_logging, timeit


@timeit
def main():
    ...


if __name__ == "__main__":
    set_logging(loglevel="info")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog="Train the model",
        description="Trains GPT model",
    )

    args = parser.parse_args()

    main()
