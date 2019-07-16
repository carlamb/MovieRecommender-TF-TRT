import tensorflow.python.keras.backend as K
import math
from movierec.model import _dcg_hr, discounted_cumulative_gain, hit_rate, MovierecModel, RankLayer
import numpy as np
import tensorflow as tf
from unittest import TestCase

TEST_PARAMS = {
    # model:
    "num_users": 5,
    "num_items": 10,
    "layers_sizes": [6, 4],
    "layers_l2reg": [0.01, 0.01],

    # training:
    "optimizer": "adam",
    "lr": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,

    "batch_size": 35,
    "num_negs_per_pos": 4,
    "k": 4
}


class TestModel(TestCase):

    def test_wrong_layers(self):
        params = TEST_PARAMS.copy()
        # add one more layer to "layers_sizes", but not to "layers_l2reg" to generate error
        params["layers_sizes"].append(2)
        self.assertRaises(ValueError, MovierecModel, params)

    def test_missing_param(self):
        params = TEST_PARAMS.copy()
        del params["num_users"]
        self.assertRaises(KeyError, MovierecModel, params)

    def test_build_mlp_model(self):
        model = MovierecModel(TEST_PARAMS).model

        self.assertEqual(model.input_shape, [(None, 1), (None, 1)])
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(len(model.layers), 9)  # 2 inputs, 2 flatten, 2 embeddings, 1 concat, 1 hidden, 1 output
        self.assertEqual(len(model.outputs), 1)
        self.assertEqual(model.output_shape, (None, 1))

        self.assertTrue(model.trainable)
        self.assertEqual(len(model.trainable_weights), 6)  # 1x2 embedding, 2(kernel+bias)x1 hidden, 2x1 output
        self.assertEqual(len(model.trainable_variables), 6)  # 1x2 embedding, 2(kernel+bias)x1 hidden, 2x1 output
        self.assertEqual(len(model.non_trainable_weights), 0)
        self.assertEqual(len(model.non_trainable_variables), 0)

    def test_not_implemented_optimizer(self):
        params = TEST_PARAMS.copy()
        params["optimizer"] = "other"
        self.assertRaises(NotImplementedError, MovierecModel, params)

    def _eval_tensor(self, tensor):
        sess = tf.Session()
        with sess.as_default():
            init = tf.global_variables_initializer()
            sess.run(init)
            return tensor.eval()

    def test_hit_rate(self):
        # last elem in each row is expected label (should be max in pred)
        y_true = np.array([[0, 0, 0, 1],
                           [0, 0, 0, 1]])
        y_pred = np.array([[0.1, 0.2, 0.9, 0.5],
                           [0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
        num_negs_per_pos = 3

        k_to_hr = {
            # for top 0, top 1: no hit: dcg=0
            0: 0.0,
            1: 0.0,
            # top 2, top 3: first row is hit, second is not
            2: 0.5,
            3: 0.5,
            # k = 4: all in top
            4: 1
        }
        for k, expected_hr in k_to_hr.items():
            avg_hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=k))
            self.assertAlmostEqual(avg_hr, expected_hr, msg="Test for k={}".format(k))

    def _dcg_index(self, index):
        return math.log(2) / math.log(index + 2)

    def test_dcg(self):
        # last elem in each row is expected label (should be max in pred)
        y_true = np.array([[0, 0, 0, 1],
                           [0, 0, 0, 1]])
        y_pred = np.array([[0.1, 0.2, 0.9, 0.5],
                           [0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
        num_negs_per_pos = 3

        # compute individual DCG. In first row label is ranked 1 (0-indexed) and in row 2, 3
        dcg0 = self._dcg_index(1)
        dcg1 = self._dcg_index(3)

        k_to_dcg = {
            # for top 0, top 1: no hit: dcg=0
            0: 0.0,
            1: 0.0,
            # top 2, top 3: first row is hit, second is not
            2: dcg0 / 2.,
            3: dcg0 / 2.,
            # k = 4: all in top
            4: (dcg0 + dcg1) / 2.
        }
        for k, expected_dcg in k_to_dcg.items():
            dcg = self._eval_tensor(discounted_cumulative_gain(y_true, y_pred, num_negs_per_pos, k=k))
            self.assertAlmostEqual(dcg, expected_dcg, msg="Test for k={}".format(k))

    def test_dgc_hr_on_ties(self):
        # last elem in each row is expected label (should be max in pred)
        y_true = np.array([[0, 0, 0, 1]])
        y_pred = np.array([[0.9, 0.55, 0.55, 0.55]], dtype=np.float32)
        num_negs_per_pos = 3

        # when there are ties, both DCG and Hit Rate should perform the same method (include or exclude)

        # for top 0, top 1: no hit
        # top 2, top 3: is a tie, but since index is last, gets excluded from top K
        for k in (0, 1, 2, 3):
            dcg, hr = _dcg_hr(y_true, y_pred, num_negs_per_pos, k=k)
            dcg = self._eval_tensor(dcg)
            hr = self._eval_tensor(hr)
            self.assertEqual(dcg, 0.0, msg="Test dcg for k={}".format(k))
            self.assertEqual(hr, 0.0, msg="Test hr for k={}".format(k))

        # only top 4 should count the hit
        dcg, hr = _dcg_hr(y_true, y_pred, num_negs_per_pos, k=4)
        dcg = self._eval_tensor(dcg)
        hr = self._eval_tensor(hr)
        self.assertAlmostEqual(dcg, self._dcg_index(3), msg="Test dcg for k={}".format(k))
        self.assertAlmostEqual(hr, 1.0, msg="Test hr for k={}".format(k))

    def test_rank_layer(self):
        rank_layer = RankLayer(num_negs_per_pos_train=2, num_negs_per_pos_eval=3, name="rank")

        # test 'training' learning phase
        K.set_learning_phase(1)

        # 2 users, 3 items per user (2 negs + 1 positive)
        input = np.array([0.9, 0.8, 0.7, 0.7, 0.8, 0.9])
        expected_rank = np.array([[0, 1, 2],
                                  [2, 1, 0]])
        pred_rank = self._eval_tensor(rank_layer.call(input))
        np.testing.assert_equal(pred_rank, expected_rank)


        # test 'eval' learning phase
        K.set_learning_phase(0)

        # 2 users, 4 items per user (3 negs + 1 positive)
        input = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.9, 0.9])
        expected_rank = np.array([[0, 1, 2, 3],
                                  [1, 2, 3, 0]])
        pred_rank = self._eval_tensor(rank_layer.call(input))
        np.testing.assert_equal(pred_rank, expected_rank)
