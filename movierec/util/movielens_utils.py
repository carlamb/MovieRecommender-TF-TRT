""" Movielens constants and utility methods """

import logging
import numpy as np
import os
import pandas as pd
import tempfile
from urllib.request import urlretrieve
import zipfile


# more info http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
MOVIELENS_URL_FORMAT = 'http://files.grouplens.org/datasets/movielens/{}.zip'
ZIP_EXTENSION = '.zip'

ML_100K = 'ml-100k'
ML_1M = 'ml-1m'
ML_20M = 'ml-20m'
MOVIELENS_DATASET_NAMES = [ML_100K, ML_1M, ML_20M]

RATINGS_FILE_NAME = {
    ML_100K: 'u.data',
    ML_1M: 'ratings.dat',
    ML_20M: 'ratings.csv'
}

MOVIES_FILE_NAME = {
    ML_100K: 'u.item',
    ML_1M: 'movies.dat',
    ML_20M: 'movies.csv'
}

SEPARATOR = {
    ML_100K: '\t|\\|',
    ML_1M: '::',
    ML_20M: ','
}

HAS_HEADER = {
    ML_100K: False,
    ML_1M: False,
    ML_20M: True
}

NUM_USERS = {
    ML_100K: 943,
    ML_1M: 6040,
    ML_20M: 138493
}

NUM_ITEMS = {
    ML_100K: 1682,
    ML_1M: 3952,
    ML_20M: 27278
}


def get_path(data_dir, dataset_name, file_name):
    return os.path.join(data_dir, dataset_name, file_name)


def get_movies_path(data_dir, dataset_name):
    return get_path(data_dir, dataset_name, MOVIES_FILE_NAME[dataset_name])


def get_ratings_path(data_dir, dataset_name):
    return get_path(data_dir, dataset_name, RATINGS_FILE_NAME[dataset_name])


def _get_file_path_download_or_raise(data_dir, dataset_name, file_name, download=True):
    # file to load: download or raise error if it does not exist
    file_path = get_path(data_dir, dataset_name, file_name)
    if not os.path.exists(file_path):
        if not download:
            raise FileNotFoundError('{} not found. Download the dataset first or set param download=True.'
                                    .format(file_path))
        else:
            download_movielens(dataset_name, data_dir)
            if not os.path.exists(file_path):
                raise FileNotFoundError('Unexpected error: {} not found after calling "download_movielens". '
                                        .format(file_path))
    return file_path


def load_movies_data(data_dir, dataset_name, col_item_id='itemId', col_movie_title='movieTitle', download=True):

    movies_file_path = _get_file_path_download_or_raise(data_dir, dataset_name, MOVIES_FILE_NAME[dataset_name], download)

    # Load dataset in a dataframe
    # Since movielens datasets are relatively small, load and manage all in Pandas. Otherwise, this could be optimized.
    movies_df = pd.read_csv(filepath_or_buffer=movies_file_path,
                            sep=SEPARATOR[dataset_name],
                            header=0 if HAS_HEADER[dataset_name] else None,
                            engine='python',  # set python engine to use regex separators
                            # only use first 2 columns:
                            usecols=(0, 1),
                            names=(col_item_id, col_movie_title),
                            dtype={col_item_id: np.int32})
    # Make Ids 0-indexed
    movies_df[col_item_id] = movies_df[col_item_id] - 1
    return movies_df


def load_ratings_data(data_dir, dataset_name, col_user_id='userId', col_item_id='itemId', col_rating='rating',
                      download=True):

    ratings_file_path = _get_file_path_download_or_raise(data_dir, dataset_name, RATINGS_FILE_NAME[dataset_name],
                                                         download)

    # Load dataset in a dataframe
    # Since movielens datasets are relatively small, load and manage all in Pandas. Otherwise, this could be optimized.
    ratings_df = pd.read_csv(filepath_or_buffer=ratings_file_path,
                             sep=SEPARATOR[dataset_name],
                             header=0 if HAS_HEADER[dataset_name] else None,
                             encoding='utf-8',
                             engine='python',  # set python engine to use regex separators
                             # only use 3 columns:
                             usecols=(0, 1, 2),
                             names=(col_user_id, col_item_id, col_rating),
                             dtype={col_user_id: np.int32, col_item_id: np.int32, col_rating: np.float32})

    # Users and ratings are 1-indexed. Make them 0-indexed
    ratings_df[col_user_id] = ratings_df[col_user_id] - 1
    ratings_df[col_item_id] = ratings_df[col_item_id] - 1
    return ratings_df


def download_movielens(dataset_name, output_dir):
    """
    Download and extract the specified movielens dataset to a directory.

    Parameters
    ----------
    dataset_name : str
        Movielens dataset name. Must be one of MOVIELENS_DATASET_NAMES.
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

        logging.info('Downloading Movielens {}'.format(dataset_url))
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
