"""
Pipeline for preparing dataset
"""

import argparse
import logging

from src.modules.data import shakespeare_data
from src.utils import set_logging, timeit


@timeit
def main() -> dict[str, str]:
    # Download shakespeare data
    path_raw_data = shakespeare_data.download_data()

    # Create dataset
    dict_path_dataset = shakespeare_data.encode_data(path=path_raw_data)

    return dict_path_dataset


if __name__ == "__main__":
    set_logging(loglevel="info")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog="Prepare training data",
        description="Generates data for training the GPT model",
    )

    args = parser.parse_args()

    main()
