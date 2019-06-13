""" Download dataset, preprocess data and provide utility methods to manage the dataset for training and evaluation """

import logging
import os
import tempfile
from urllib.request import urlretrieve
import zipfile

# Movielens datasets constants (detailed info: https://grouplens.org/datasets/movielens/)
MOVIELENS_URL_FORMAT = 'http://files.grouplens.org/datasets/movielens/{}.zip'
MOVIELENS_DATASET_NAMES = ['ml-100k', 'ml-1m', 'ml-20m']
ZIP_EXTENSION = '.zip'


def download_movielens(dataset_name, output_dir):
    """
    Download and extract the specified movielens dataset to a directory.

    Parameters
    ----------
    dataset_name : str
        Movielens dataset name.
    output_dir : str
        Directory where to save the downloaded dataset.

    Returns
    -------
        Dataset path (folder name will be output_dir/dataset_name)
    """
    if dataset_name not in MOVIELENS_DATASET_NAMES:
        raise ValueError('Invalid dataset name {}. Must be one of {}'
                         .format(dataset_name, ', '.join(MOVIELENS_DATASET_NAMES)))

    with tempfile.TemporaryDirectory() as temp_dir:  # automatically cleaned up after this context
        # download dataset zip file into temporary folder
        dataset_url = MOVIELENS_URL_FORMAT.format(dataset_name)
        dataset_file_name = os.path.join(temp_dir, dataset_name + ZIP_EXTENSION)
        urlretrieve(dataset_url, dataset_file_name)

        logging.info('Downloaded {}'.format(dataset_file_name))

        # unzip to output directory
        if not os.path.isdir(output_dir):
           os.makedirs(output_dir)

        with zipfile.ZipFile(dataset_file_name, "r") as zip_file:
            zip_file.extractall(output_dir)

            # movielens datasets zips contain a single top level folder with data files inside
            dataset_dir = os.path.join(output_dir, dataset_name)

            logging.info("Dataset extracted to {}".format(dataset_dir))

    return dataset_dir
