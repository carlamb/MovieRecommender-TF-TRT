""" Download dataset, preprocess data and provide utility methods to manage the dataset for training and evaluation """

import logging
import numpy as np
import os
import pandas as pd
import tempfile
from urllib.request import urlretrieve
import zipfile


# more info http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
MOVIELENS_URL_FORMAT = 'http://files.grouplens.org/datasets/movielens/{}.zip'

ML_100K = 'ml-100k'
ML_1M = 'ml-1M'
ML_20M = 'ml-20M'
MOVIELENS_DATASET_NAMES = [ML_100K, ML_1M, ML_20M]

ZIP_EXTENSION = '.zip'

RATINGS_FILE_NAME = {
    ML_100K: 'u.data',
    ML_1M: 'ratings.dat',
    ML_20M: 'ratings.csv'
}
SEPARATOR = {
    ML_100K: '\t',
    ML_1M: '::',
    ML_20M: ','
}

HAS_HEADER = {
    ML_100K: False,
    ML_1M: False,
    ML_20M: True
}

COL_USER_ID = 'userId'
COL_ITEM_ID = 'itemId'
COL_RATING = 'rating'


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


def load_ratings_train_test_sets(dataset_name, data_dir, download=True):
    """
    Load Movielens ratings dataset from a file. Optionally download it first if it does not exist.
    Split the dataset in train and test sets.

    Parameters
    ----------
    dataset_name : str
        Movielens dataset name. Must be one of MOVIELENS_DATASET_NAMES.
    data_dir : str or os.path
        Dataset directory to read from. The file to read from the directory will be obtained from RATINGS_FILE_NAME.
    download : boolean
        Download and extract Movielens dataset if it does not exist in the 'data_dir'. Default=True.

    Returns
    -------
        (train, test) : (DataFrame)
            Dataset split into 2 DataFrames with columns COL_USER_ID, COL_ITEM_ID, COL_RATING.
    """

    if dataset_name not in MOVIELENS_DATASET_NAMES:
        raise ValueError('Invalid dataset name {}. Must be one of {}'
                         .format(dataset_name, ', '.join(MOVIELENS_DATASET_NAMES)))

    # file to load: download or raise error if it does not exist
    ratings_file_path = os.path.join(data_dir, RATINGS_FILE_NAME[dataset_name])
    if not os.path.exists(ratings_file_path):
        if not download:
            raise FileNotFoundError('{} not found. Download the dataset first or set param download=True.'
                                    .format(ratings_file_path))
        else:
            download_movielens(dataset_name, data_dir)
            if not os.path.exists(ratings_file_path):
                raise FileNotFoundError('Unexpected error: {} not found after calling "download_movielens". '
                                        .format(ratings_file_path))

    # Load dataset in a dataframe
    ratings_df = pd.read_csv(filepath_or_buffer=ratings_file_path,
                             sep=SEPARATOR[dataset_name],
                             header=0 if HAS_HEADER[dataset_name] else None,
                             encoding='utf-8',
                             engine='python',  # set python engine to use regex separators
                             # only use 3 columns:
                             usecols=(0, 1, 2),
                             names=(COL_USER_ID, COL_ITEM_ID, COL_RATING),
                             dtype={COL_USER_ID: np.int32, COL_ITEM_ID: np.int32, COL_ITEM_ID: np.float32})

    # TODO: log a small summary, check for duplicates?

    # TODO: implement other splitting options. Add more params for splits
    # split in train and test by taking the out the first rating of every user as test, the rest is train
    grouped_by_user = ratings_df.groupby(COL_USER_ID, group_keys=False)
    test_ratings_df = grouped_by_user.apply(lambda x: x.head(1))
    train_ratings_df = grouped_by_user.apply(lambda x: x.iloc[1:])

    # reset the indexes as they are kept from the original dataframe
    test_ratings_df = test_ratings_df.reset_index(drop=True)
    train_ratings_df = train_ratings_df.reset_index(drop=True)

    return train_ratings_df, test_ratings_df
