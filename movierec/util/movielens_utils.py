
import numpy as np
import os
import pandas as pd


# more info http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
MOVIELENS_URL_FORMAT = 'http://files.grouplens.org/datasets/movielens/{}.zip'

ML_100K = 'ml-100k'
ML_1M = 'ml-1m'
ML_20M = 'ml-20m'
MOVIELENS_DATASET_NAMES = [ML_100K, ML_1M, ML_20M]

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


def get_ratings_path(data_dir, dataset_name):
    return os.path.join(data_dir, dataset_name, RATINGS_FILE_NAME[dataset_name])


def load_ratings_data(data_dir, dataset_name, col_user_id='userId', col_item_id='itemId', col_rating='rating'):

    ratings_file_path = get_ratings_path(data_dir, dataset_name)

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
