"""
Utility functions
"""
import codecs
import contextlib
import copy
import csv
import datetime
import glob
import logging
import os
import sys
import time
import zipfile

import joblib
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def timeit(f):
    """
    A simple decorator for measuring time of a function call
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Measuring time for {f}")
        res = f(*args, **kwargs)
        end = time.time()
        logger.info(f"{f} took {end - start:.8f} seconds")
        return res

    return wrapper


def read_config(file_path):
    """
    Read a yaml file
    """
    with open(file_path, encoding="utf8") as f:
        config = yaml.full_load(f)
    return config


def get_timestamp() -> str:
    """
    Get timestamp in YYYYMMDD-hhmm format
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def gen_path(path_dir, fname, timestamp=True):
    """
    Generate a path to save something

    Args:
        path_dir (str): Path to the directory
        fname (str): Filename of the object to save
        timestamp (bool): Whether to prefix filename with the timestamp

    Returns:
        (str) Full path
    """
    os.makedirs(path_dir, exist_ok=True)

    if timestamp:
        path_save = os.path.join(path_dir, "{}_{}".format(get_timestamp(), fname))
    else:
        path_save = os.path.join(path_dir, fname)

    return path_save


def save_parquet(df, path_dir, fname, timestamp=True):
    """
    Save a parquet file
    """
    path_save = gen_path(path_dir, fname, timestamp)
    logger.info("Saving to {}".format(str(path_save)))

    df.to_parquet(path_save)

    return path_save


def save_joblib_dump(value, path_dir, fname, timestamp=True):
    """
    Save a pkl file with joblib
    """
    path_save = gen_path(path_dir, fname, timestamp)
    logger.info("Saving to {}".format(str(path_save)))
    joblib.dump(value, filename=path_save)
    return path_save


def save_excel(df, path_dir, fname, timestamp=True, verbose=True) -> str:
    """
    Save a DataFrame as an excel
    """
    path_save = gen_path(path_dir, fname, timestamp)
    if verbose:
        logger.info("Saving to {}".format(str(path_save)))
    df.to_excel(path_save, index=False)
    return path_save


def save_csv(df, path_dir, fname, timestamp=True, verbose=True) -> str:
    """
    Save a DataFrame as an csv
    """

    path_save = gen_path(path_dir, fname, timestamp)
    if verbose:
        logger.info("Saving to {}".format(str(path_save)))
    df.to_csv(path_save, index=False, quoting=csv.QUOTE_ALL)

    return path_save


def read_parquets(path_glob) -> list:
    """
    Read all parquets provided in a glob path

    Args:
        path_glob (str): Glob-like path string

    Returns:
        (list) A list of pandas DataFrames
    """
    list_paths = glob.glob(path_glob)
    logger.info("Reading {} parquet files".format(len(list_paths)))

    list_parquets = []
    for path in tqdm(list_paths):
        list_parquets.append(pd.read_parquet(path))

    return list_parquets


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class ColoredStreamHandler(logging.StreamHandler):
    # From https://pod.hatenablog.com/entry/2020/03/01/221715
    # Also based on https://github.com/horoiwa/ProjectBoilerPlate/blob/main/project/package/constants.py
    cmap = {
        "TRACE": "[TRACE]",
        "DEBUG": "\x1b[0;36mDEBUG\x1b[0m",
        "INFO": "\x1b[0;32mINFO\x1b[0m",
        "WARNING": "\x1b[0;33mWARN\x1b[0m",
        "WARN": "\x1b[0;33mwWARN\x1b[0m",
        "ERROR": "\x1b[0;31mERROR\x1b[0m",
        "ALERT": "\x1b[0;37;41mALERT\x1b[0m",
        "CRITICAL": "\x1b[0;37;41mCRITICAL\x1b[0m",
    }

    def emit(self, record: logging.LogRecord) -> None:
        record = copy.deepcopy(record)
        record.levelname = self.cmap[record.levelname]
        super().emit(record)


def set_logging(loglevel: str) -> None:
    """Setup for logging

    Args:
        loglevel (str): log level
    """
    log_map = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    stream_handler = ColoredStreamHandler(stream=sys.stdout)

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s {%(filename)s:%(lineno)d} - %(message)s",
        level=log_map.get(loglevel, logging.INFO),
        handlers=[stream_handler],
    )


def banner(text, ch="#", length=78, logger=None):
    """
    Simple function to print a banner
    """
    spaced_text = f" {text} "
    banner = spaced_text.center(length, ch)
    if logger:
        logger.info(ch.center(length, ch))
        logger.info(banner)
        logger.info(ch.center(length, ch))
    else:
        print(ch.center(length, ch))
        print(banner)
        print(ch.center(length, ch))


def read_csv(
    full_csv_path: str, columns: list[str], delimiter: str = ","
) -> pd.DataFrame:
    """
    read csv
    """
    try:
        df = pd.read_csv(
            full_csv_path, delimiter=delimiter, usecols=columns, low_memory=False
        )
    except UnicodeDecodeError:
        with codecs.open(full_csv_path, "r", "Shift-JIS", "ignore") as file:
            df = pd.read_table(
                file, delimiter=delimiter, usecols=columns, low_memory=False
            )
    return df


def unzip_files(zip_path: str, dir_save: str) -> list[str]:
    """
    unzip files
    Args:
        zip_path (str) : full path of a zipfile
        dir_save (str) : full path to save
    Returns:
        csv_file_names (list[str]): The list of unzipped file names (NOT FULL PATH)
    """

    if zip_path.endswith(".zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dir_save)
            csv_file_names = zip_ref.namelist()
    # for csv file
    elif zip_path.endswith(".csv"):
        csv_file_names = list(zip_path)

    return csv_file_names


def remove_rows(df: pd.DataFrame, column_name: str, words: list[str]) -> pd.DataFrame:
    for word in words:
        df = df[~df[column_name].str.contains(word, case=False)]
    return df


def flatten(list_of_lists):
    """
    Flatten a list of lists to a combined list
    Args:
        list_of_lists (list): List of lists

    Returns:
        (list) Flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]


def chunker(iterable, total_length, chunksize):
    """
    Batch an iterable and return a generator
    """
    return (
        iterable[pos : pos + chunksize] for pos in range(0, total_length, chunksize)
    )
