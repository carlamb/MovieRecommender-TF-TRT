from movierec.data_pipeline import load_ratings_train_test_sets, MovieLensDataGenerator

from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
import pandas as pd


class TestDataPipeline(TestCase):

    def test_wrong_database_name_load(self):
        self.assertRaises(ValueError, load_ratings_train_test_sets, 'wrong db', '/tmp/')

    @patch('movierec.data_pipeline.os.path.exists')
    def test_file_not_found(self, mock_path_exists):
        mock_path_exists.return_value = False
        self.assertRaises(FileNotFoundError, load_ratings_train_test_sets, 'ml-100k', 'MOCK TEST PATH', False)

    @patch('movierec.data_pipeline.pd.read_csv')
    @patch('movierec.data_pipeline.os.path.exists')
    def test_load_ratings_train_test_sets(self, mock_path_exists, mock_read_csv):
        mock_path_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'userId': pd.Series([1, 1, 1, 1, 2, 2, 2]),
                                                   'itemId': pd.Series([111, 123, 200, 333, 101, 222, 300]),
                                                   'rating': pd.Series([5., 5., 4., 3., 2., 4., 5.])})
        train, validation, test = load_ratings_train_test_sets('ml-100k', 'ml-100k')

        expected_train = pd.DataFrame({'userId': pd.Series([1, 1, 2]),
                                       'itemId': pd.Series([111, 123, 101]),
                                       'rating': pd.Series([5., 5., 2])})

        expected_validation = pd.DataFrame({'userId': pd.Series([1, 2]),
                                            'itemId': pd.Series([200, 222]),
                                            'rating': pd.Series([4., 4.])})

        expected_test = pd.DataFrame({'userId': pd.Series([1, 2]),
                                      'itemId': pd.Series([333, 300]),
                                      'rating': pd.Series([3., 5.])})
        pd.testing.assert_frame_equal(train, expected_train)
        pd.testing.assert_frame_equal(validation, expected_validation)
        pd.testing.assert_frame_equal(test, expected_test)

    @patch('movierec.data_pipeline.download_movielens')
    @patch('movierec.data_pipeline.pd.read_csv')
    @patch('movierec.data_pipeline.os.path.exists')
    def test_load_ratings_train_test_sets_download(self, mock_path_exists, mock_read_csv, mock_download_movielens):
        # prepare mocks

        mock_path_exists.side_effect = [False, True]  # first return False, then True
        mock_read_csv.return_value = pd.DataFrame({'userId': pd.Series([1, 1, 1, 2, 2, 2]),
                                                   'itemId': pd.Series([100, 111, 200, 222, 300, 100]),
                                                   'rating': pd.Series([4., 5., 5., 4., 3., 2.])})
        # call tested method
        load_ratings_train_test_sets('ml-100k', 'testPath')

        # verify mocks are called (outputs verified in another test)
        mock_download_movielens.assert_called_with('ml-100k', 'testPath')
        mock_read_csv.assert_called_once()

    @patch('movierec.data_pipeline.MovieLensDataGenerator.num_items', new_callable=PropertyMock)
    def test_generator_get_item(self, mock_num_items):
        # mock data with 2 users and 5 items
        mock_num_items.return_value = 5
        data = pd.DataFrame({'userId': pd.Series([0, 0, 0, 1]),
                             'itemId': pd.Series([0, 1, 2, 3]),
                             'rating': pd.Series([5., 5., 4., 3.])})
        extra_data = pd.DataFrame({'userId': pd.Series([0, 1]),
                                   'itemId': pd.Series([4, 4]),
                                   'rating': pd.Series([3., 2.])})

        test_class = MovieLensDataGenerator('ml-100k', data, batch_size=6, negatives_per_positive=2,
                                            extra_data_df=extra_data, shuffle=False)

        # test first batch:
        # size:6, positives:2, negatives:4 (2 negatives_per_positive *2 positives)
        for _ in range(10):  # test 5 times because there is random component
            batch_x, batch_y = test_class[0]
            self.assertEqual(len(batch_x), 2)
            x_users = batch_x[0]
            x_items = batch_x[1]

            # First batch has only user: 0
            np.testing.assert_equal(x_users, np.zeros(6, dtype=np.int))
            # Items for user 0: [0,1,2] in data, and [4] in extra data
            # There are 5 items, so the only option in this batch is that all negative items are: 3
            np.testing.assert_equal(x_items, np.array([0, 1, 3, 3, 3, 3]))
            np.testing.assert_equal(batch_y, np.array([1, 1, 0, 0, 0, 0]))

        # test second batch:
        # size:6, positives:2, negatives:4 (2 negatives_per_positive *2 positives)
        for _ in range(10):  # test 5 times because there is random component
            batch_x, batch_y = test_class[1]
            self.assertEqual(len(batch_x), 2)
            x_users = batch_x[0]
            x_items = batch_x[1]

            # Second batch has users: 0, 1
            np.testing.assert_equal(x_users, np.array([0, 1, 0, 0, 1, 1]))

            # assert positives and negatives of user 0 (only '3' is possible for user 0)
            np.testing.assert_equal(x_items[:4], np.array([2, 3, 3, 3]))
            np.testing.assert_equal(batch_y[:4], np.array([1, 1, 0, 0]))

            # assertions for negatives of user 1. Only items [0, 1, 2] are possible
            # and there must be no repeated items (2 chosen among 3 options, repetitions only happen if no
            # other option is possible)
            self.assertEqual(len(np.setdiff1d(np.array([0, 1, 2]), x_items[4:])), 1)
            np.testing.assert_equal(batch_y[4:], np.array([0, 0]))
