from movierec.util.movielens_utils import load_ratings_data

from unittest import TestCase
from unittest.mock import patch

import pandas as pd


class TestMovielensUtils(TestCase):

    @patch('movierec.util.movielens_utils.os.path.exists')
    def test_file_not_found(self, mock_path_exists):
        mock_path_exists.return_value = False
        self.assertRaises(FileNotFoundError, load_ratings_data, 'MOCK TEST PATH', 'ml-100k', download=False)

    @patch('movierec.util.movielens_utils.download_movielens')
    @patch('movierec.util.movielens_utils.pd.read_csv')
    @patch('movierec.util.movielens_utils.os.path.exists')
    def test_load_ratings_data_download(self, mock_path_exists, mock_read_csv, mock_download_movielens):
        # prepare mocks

        mock_path_exists.side_effect = [False, True]  # first return False, then True
        mock_read_csv.return_value = pd.DataFrame({'userId': pd.Series([1, 1, 1, 2, 2, 2]),
                                                   'itemId': pd.Series([100, 111, 200, 222, 300, 100]),
                                                   'rating': pd.Series([4., 5., 5., 4., 3., 2.])})
        # call tested method
        load_ratings_data('testPath', 'ml-100k', download=True)

        # verify mocks are called (outputs verified in another test)
        mock_download_movielens.assert_called_with('ml-100k', 'testPath')
        mock_read_csv.assert_called_once()

    @patch('movierec.util.movielens_utils.pd.read_csv')
    @patch('movierec.util.movielens_utils.os.path.exists')
    def test_load_ratings_data(self, mock_path_exists, mock_read_csv):
        mock_path_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'userId': pd.Series([1, 1, 1, 1, 2, 2, 2]),
                                                   'itemId': pd.Series([111, 123, 200, 333, 101, 222, 300]),
                                                   'rating': pd.Series([5., 5., 4., 3., 2., 4., 5.])})
        ratings = load_ratings_data('test_data', 'ml-100k')

        # expected values (0-indexed)
        expected_ratings_df = pd.DataFrame({'userId': pd.Series([0, 0, 0, 0, 1, 1, 1]),
                                            'itemId': pd.Series([110, 122, 199, 332, 100, 221, 299]),
                                            'rating': pd.Series([5., 5., 4., 3., 2., 4., 5.])})

        pd.testing.assert_frame_equal(ratings, expected_ratings_df)
