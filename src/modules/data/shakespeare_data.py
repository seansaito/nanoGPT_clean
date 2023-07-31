"""
Based on https://github.com/karpathy/nanoGPT/tree/master/data

"""

import logging
import pickle

import numpy as np
import requests
import tiktoken

from src import DATA_DIR
from src.utils import gen_path

SHAKESPEARE_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_NAME = "tiny_shakespeare"

logger = logging.getLogger(__name__)


def download_data(url: str = SHAKESPEARE_DATA_URL) -> str:
    """
    Download shakespeare data
    """

    logger.info("Fetching data from {}".format(url))
    response = requests.get(url).text

    path_raw_input = gen_path(
        path_dir=DATA_DIR / "input" / DATASET_NAME, fname="input.txt", timestamp=False
    )
    logger.info("Writing to {}".format(path_raw_input))
    with open(path_raw_input, "w") as fp:
        fp.write(response)

    return path_raw_input


def encode_data(path: str, train_ratio: float = 0.9) -> dict[str, str]:
    """
    Encode the data and return the files
    """
    # Read the data
    with open(path, "r") as fp:
        data = fp.read()

    # Split data into train and test
    num_lines = len(data)
    train_data = data[: int(num_lines * train_ratio)]
    val_data = data[int(num_lines * (1 - train_ratio)) :]

    # Encode using tiktoken GPT-2 BPE
    encoder = tiktoken.get_encoding("gpt2")
    train_ids = encoder.encode_ordinary(train_data)
    val_ids = encoder.encode_ordinary(val_data)

    logger.info(f"Train has {len(train_ids):,} tokens")
    logger.info(f"Test has {len(val_ids):,} tokens")

    # Export
    arr_train_ids = np.array(train_ids, dtype=np.uint16)
    arr_val_ids = np.array(val_ids, dtype=np.uint16)

    path_train_save = gen_path(
        path_dir=DATA_DIR / "dataset" / DATASET_NAME, fname="train.bin", timestamp=False
    )

    path_val_save = gen_path(
        path_dir=DATA_DIR / "dataset" / DATASET_NAME, fname="val.bin", timestamp=False
    )

    logger.info("Saving training data to: {}".format(path_train_save))
    arr_train_ids.tofile(path_train_save)

    logger.info("Saving validation data to: {}".format(path_val_save))
    arr_val_ids.tofile(path_val_save)

    # Save metadata
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    dict_meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    path_meta_data = gen_path(
        path_dir=DATA_DIR / "dataset" / DATASET_NAME, fname="meta.pkl", timestamp=False
    )
    with open(path_meta_data, "wb") as fp:
        pickle.dump(dict_meta, fp)

    return {
        "train_ids": path_train_save,
        "val_ids": path_val_save,
        "meta_data": path_meta_data,
    }
