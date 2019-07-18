""" Download dataset, preprocess data and provide utility methods to manage the dataset for training and evaluation """

import logging
import numpy as np
import os
import tempfile
from tensorflow.python.keras.utils import Sequence
from urllib.request import urlretrieve
import util.movielens_utils as ml
import zipfile


ZIP_EXTENSION = '.zip'

COL_USER_ID = 'userId'
COL_ITEM_ID = 'itemId'
COL_RATING = 'rating'
COL_LABEL = 'label'


class MovieLensDataGenerator(Sequence):

    def __init__(self, dataset_name, data_df, batch_size, negatives_per_positive, extra_data_df=None, shuffle=True):
        # TODO yield last incomplete batch (optional)
        """
        Create a generator that provides batches of Movielens data for training or testing purposes.
        Every batch contains positive and negative examples. The positive examples are taken from `data_df`.
        The negative examples are generated for every batch by picking a random list of items per user that are not
        present in either `data_df` or `extra_data_df`.

        Parameters
        ----------
        dataset_name : str
            Movielens dataset name. Must be one of MOVIELENS_DATASET_NAMES.
        data_df : `Dataframe`
            Data to build the generator from. It is a Movielens dataset containing users, items and ratings.
        batch_size : int
            Batch size to yield data. Must be divisible by (negatives_per_positive + 1)
        negatives_per_positive : int
            Number of negatives examples to generate for every positive example in a batch. Batch size must be divisible
            by (negatives_per_positive + 1).
        extra_data_df : `DataFrame`
            Optional dataframe to be used when computing negatives. Negative items for a user are those that do not
            exist for that user in 'data_df' or 'extra_data_df'. The data of this is not directly provided by the
            generator.
        shuffle : bool
            Whether to shuffle the data_df between epochs. Note that the negative examples are randomly generated for
            every batch, so, even when `shuffle` is False, the batches will be different every time (positives will be
            equal, but negatives will be different).
        """

        if dataset_name not in ml.MOVIELENS_DATASET_NAMES:
            raise ValueError('Invalid dataset name {}. Must be one of {}'
                             .format(dataset_name, ', '.join(ml.MOVIELENS_DATASET_NAMES)))

        if negatives_per_positive <= 0:
            raise ValueError("negatives_per_positive must be > 0, found {}".format(negatives_per_positive))

        if batch_size % (negatives_per_positive + 1):
            raise ValueError("Batch size must be divisible by (negatives_per_positive + 1). Found: batch_size={}, "
                             "negatives_per_positive={}".format(batch_size, negatives_per_positive))

        self._dataset_name = dataset_name
        self._num_users = ml.NUM_USERS[dataset_name]
        self._num_items = ml.NUM_ITEMS[dataset_name]
        self.data = data_df
        self.extra_data = extra_data_df
        self.batch_size = batch_size
        self.negatives_per_positive = negatives_per_positive
        self.num_positives_per_batch = self.batch_size // (negatives_per_positive + 1)
        self.num_negatives_per_batch = self.batch_size - self.num_positives_per_batch
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))

        self.on_epoch_end()
        logging.info('Created generator for {}. Num users={}, num items={}, num_batches={}, batch size={}, '
                     'positives per batch={}, negatives per batch={}'
                     .format(dataset_name, self._num_users, self._num_items, len(self), batch_size,
                             self.num_positives_per_batch, self.num_negatives_per_batch))

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def dataset_name(self):
        return self._dataset_name

    def __len__(self):
        """
        Number of batches in the Sequence.

        Returns
        -------
        The number of batches.
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def _get_random_negatives_and_positive(self, row):
        # Given a DataFrame row, generate an array of random negative items and the positive one at the end.

        user = row[COL_USER_ID]
        positives_user = self.data[self.data[COL_USER_ID] == user][COL_ITEM_ID]
        if self.extra_data is not None:
            positives_user = positives_user.append(self.extra_data[self.extra_data[COL_USER_ID] == user][COL_ITEM_ID])

        # obtain possible negatives
        possible_negs = np.setdiff1d(np.arange(self.num_items), positives_user, assume_unique=True)

        # select randomly, without replacement, if possible
        replace = len(possible_negs) < self.negatives_per_positive
        negative_items = np.random.choice(possible_negs, self.negatives_per_positive, replace=replace)
        return np.append(negative_items, int(row[COL_ITEM_ID]))

    def __getitem__(self, idx):
        """
        Generate and return a batch of data. A batch contains a ratio of 1:self.negatives_per_positive of positive
        to negative examples per user. The positive examples are taken from self.data. The negative examples are
        generated by this method by picking a random list of items per user that are not present in either `self.data`
        or `self.extra_data`. The random selection is performed without replacement, when possible (if num items to be
        selected < num possible items to select).

        Parameters
        ----------
        idx : int
            Batch index. Must be between 0 and len(self)-1.

        Returns
        -------
        X, Y: Tuple of type (List of 2 np.arrays, np.arrays)
            A batch of data. `X` are the inputs: an array of users and an array of items. `Y` is an array of labels.
            All 3 arrays have size `self.batch_size`.
        """

        # Get indexes of the positive examples for this batch:
        idxs_pos = self.indexes[idx * self.num_positives_per_batch:(idx + 1) * self.num_positives_per_batch]

        # Get the positives
        positives = self.data.iloc[idxs_pos]
        # users are repeated to include negatives
        x_user = np.repeat(positives[COL_USER_ID].values, 1 + self.negatives_per_positive)

        # items: for every positive, create array of random negatives and the positive at the end
        items_with_negatives = positives.apply(self._get_random_negatives_and_positive, axis=1)
        x_item = np.concatenate(items_with_negatives.values)

        # labels: first negative labels, then one positive (N times)
        y = np.tile([0] * self.negatives_per_positive + [1], self.num_positives_per_batch)

        return [x_user, x_item], y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


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

    if dataset_name not in ml.MOVIELENS_DATASET_NAMES:
        raise ValueError('Invalid dataset name {}. Must be one of {}'
                         .format(dataset_name, ', '.join(ml.MOVIELENS_DATASET_NAMES)))

    with tempfile.TemporaryDirectory() as temp_dir:  # automatically cleaned up after this context
        # download dataset zip file into temporary folder
        dataset_url = ml.MOVIELENS_URL_FORMAT.format(dataset_name)
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


def load_ratings_train_test_sets(dataset_name, data_dir, download=True):
    """
    Load Movielens ratings dataset from a file. Optionally download it first if it does not exist.
    Split the dataset in train, validation and test sets.

    Parameters
    ----------
    dataset_name : str
        Movielens dataset name. Must be one of MOVIELENS_DATASET_NAMES.
    data_dir : str or os.path
        Dataset directory to read from. The file to read from the directory will be:
        data_dir/dataset_name/RATINGS_FILE_NAME[dataset_name].
    download : boolean
        Download and extract Movielens dataset if it does not exist in the 'data_dir'. Default=True.

    Returns
    -------
        (train, validation, test) : (DataFrame)
            Dataset split into 3 DataFrames with columns COL_USER_ID, COL_ITEM_ID, COL_RATING.
    """

    if dataset_name not in ml.MOVIELENS_DATASET_NAMES:
        raise ValueError('Invalid dataset name {}. Must be one of {}'
                         .format(dataset_name, ', '.join(ml.MOVIELENS_DATASET_NAMES)))

    # file to load: download or raise error if it does not exist
    ratings_file_path = ml.get_ratings_path(data_dir, dataset_name)
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
    # Since movielens datasets are relatively small, load and manage all in Pandas. Otherwise, this could be optimized.
    ratings_df = ml.load_ratings_data(data_dir, dataset_name, COL_USER_ID, COL_ITEM_ID, COL_RATING)

    # TODO: log a small summary, check for duplicates?

    # TODO: implement other splitting options. Add more params for splits
    # split in train, validation and test by taking the out the first rating of every user as test, the rest is train
    # (every user in movielens has at least 20 items rated)
    grouped_by_user = ratings_df.groupby(COL_USER_ID, group_keys=False)
    test_ratings_df = grouped_by_user.apply(lambda x: x.iloc[[-1]])
    validation_df = grouped_by_user.apply(lambda x: x.iloc[[-2]])
    train_ratings_df = grouped_by_user.apply(lambda x: x.iloc[:-2])

    # reset the indexes as they are kept from the original dataframe
    test_ratings_df = test_ratings_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    train_ratings_df = train_ratings_df.reset_index(drop=True)

    return train_ratings_df, validation_df, test_ratings_df
