from movierec.data_pipeline import load_ratings_train_test_sets

from unittest import TestCase
from unittest.mock import patch

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

        expected_train = pd.DataFrame({'userId': pd.Series([1, 2, 2]),
                                      'itemId': pd.Series([200, 300, 100]),
                                       'rating': pd.Series([5., 3., 2])})

