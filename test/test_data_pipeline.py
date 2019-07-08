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

        # expected values (0-indexed)
        expected_train = pd.DataFrame({'userId': pd.Series([0, 0, 1]),
                                       'itemId': pd.Series([110, 122, 100]),
                                       'rating': pd.Series([5., 5., 2])})

        expected_validation = pd.DataFrame({'userId': pd.Series([0, 1]),
                                            'itemId': pd.Series([199, 221]),
                                            'rating': pd.Series([4., 4.])})

        expected_test = pd.DataFrame({'userId': pd.Series([0, 1]),
                                      'itemId': pd.Series([332, 299]),
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
        # mock data with 3 users and 5 items
        mock_num_items.return_value = 5
        data = pd.DataFrame({'userId': pd.Series([0, 2, 0, 1]),
                             'itemId': pd.Series([0, 0, 2, 3])})
        extra_data = pd.DataFrame({'userId': pd.Series([0, 0, 1, 2, 2, 2]),
                                   'itemId': pd.Series([1, 4, 4, 1, 2, 4])})

        test_class = MovieLensDataGenerator('ml-100k', data, batch_size=6, negatives_per_positive=2,
                                            extra_data_df=extra_data, shuffle=False)

        # test first batch:
        # size:6, positives:2, negatives:4 (2 negatives_per_positive *2 positives)
        for _ in range(10):  # test 10 times because there is a random component
            batch_x, batch_y = test_class[0]
            self.assertEqual(len(batch_x), 2)
            x_users = batch_x[0]
            x_items = batch_x[1]

            # Second batch has users: 0, 2
            np.testing.assert_equal(x_users, np.array([0, 0, 0, 2, 2, 2]))
            # Items for user 0: [0,2] in data, and [1,4] in extra data, user 2: [0] in data, [1,2,4] in extra_data
            # There are 5 items, so the only option in this batch is that all negative items are: 3
            np.testing.assert_equal(x_items, np.array([3, 3, 0, 3, 3, 0]))
            np.testing.assert_equal(batch_y, np.array([0, 0, 1, 0, 0, 1]))

        # test second batch:
        # size:6, positives:2, negatives:4 (2 negatives_per_positive *2 positives)
        for _ in range(10):  # test 10 times because there is a random component
            batch_x, batch_y = test_class[1]
            self.assertEqual(len(batch_x), 2)
            x_users = batch_x[0]
            x_items = batch_x[1]

            # Second batch has users: 0, 1
            np.testing.assert_equal(x_users, np.array([0, 0, 0, 1, 1, 1]))

            # verify user 0 (only '3' is possible negative)
            np.testing.assert_equal(x_items[:3], np.array([3, 3, 2]))
            np.testing.assert_equal(batch_y[:3], np.array([0, 0, 1]))

            # verify user 1. Only items [0, 1, 2] are possible
            # and there must be no repeated items (2 chosen among 3 options, repetitions only happen if no
            # other option is possible)
            self.assertEqual(len(np.setdiff1d(np.array([0, 1, 2]), x_items[3:5])), 1)
            self.assertEqual(3, x_items[5])
            np.testing.assert_equal(batch_y[3:], np.array([0, 0, 1]))

    @patch('movierec.data_pipeline.MovieLensDataGenerator.num_items', new_callable=PropertyMock)
    def test_generator_get_item_duplicated_user_batch(self, mock_num_items):
        # mock data with 2 users and 6 items
        mock_num_items.return_value = 6
        data = pd.DataFrame({'userId': pd.Series([0, 0, 0, 1]),
                             'itemId': pd.Series([0, 1, 2, 3]),
                             'rating': pd.Series([5., 5., 4., 3.])})

        test_class = MovieLensDataGenerator('ml-100k', data, batch_size=6, negatives_per_positive=2,
                                            extra_data_df=None, shuffle=False)

        not_always_equals_in_batch = False
        not_always_equals_between_runs = False
        last_run_array = None

        # test first batch:
        # size:6, positives:2, negatives:4 (2 negatives_per_positive *2 positives)
        for _ in range(50):  # test 50 times because there is a random component
            batch_x, batch_y = test_class[0]
            self.assertEqual(len(batch_x), 2)
            x_users = batch_x[0]
            x_items = batch_x[1]

            # First batch has only user: 0
            np.testing.assert_equal(x_users, np.zeros(6, dtype=np.int))
            # Items for user 0: [0,1,2]. Possible negatives are [3, 4, 5]
            # There must be no repeated items (2 chosen among 3 options, repetitions only happen if no
            # other option is possible)
            self.assertEqual(len(np.setdiff1d(np.array([3, 4, 5]), x_items[:2])), 1)
            self.assertEqual(len(np.setdiff1d(np.array([3, 4, 5]), x_items[3:6])), 1)
            np.testing.assert_equal(batch_y, np.array([0, 0, 1, 0, 0, 1]))

            # verify that generated negatives are not always the same
            if not np.array_equal(x_items[:2], x_items[3:6]):
                not_always_equals_in_batch = True
            if last_run_array is not None and not np.array_equal(last_run_array, x_items[:2]):
                not_always_equals_between_runs = True
            last_run_array = x_items[:2]

        self.assertTrue(not_always_equals_between_runs,
                        "Randomly generated negatives should not always be equals between runs")
        self.assertTrue(not_always_equals_in_batch,
                        "Randomly generated negatives should not always be equals in the same batch")

    def test_generator_value_errors(self):
        data = pd.DataFrame({'userId': pd.Series([0, 0, 0, 1]),
                             'itemId': pd.Series([0, 1, 2, 3]),
                             'rating': pd.Series([5., 5., 4., 3.])})

        with self.assertRaisesRegex(ValueError, 'Invalid dataset name'):
            MovieLensDataGenerator('wrong_name', data, batch_size=6, negatives_per_positive=2)
        with self.assertRaisesRegex(ValueError, 'negatives_per_positive must be > 0'):
            MovieLensDataGenerator('ml-100k', data, batch_size=6, negatives_per_positive=0)
        with self.assertRaisesRegex(ValueError, 'Batch size must be divisible by'):
            MovieLensDataGenerator('ml-100k', data, batch_size=10, negatives_per_positive=6)
