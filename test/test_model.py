import math
from movierec.model import MovierecModel, hit_rate, dcg
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
        self.assertEqual(len(model.layers), 7)  # 2 inputs, 2 embeddings, 1 concat, 1 hidden, 1 output
        self.assertEqual(len(model.outputs), 1)
        self.assertEqual(model.output_shape, (None, 1, 1))

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

        # for top 0, top 1: hr=0
        hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=0))
        self.assertEqual(hr, 0.0)
        hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=1))
        self.assertEqual(hr, 0.0)

        # top 2, top 3: first row is hit, second is not
        hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=2))
        self.assertEqual(hr, 0.5)
        hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=3))
        self.assertEqual(hr, 0.5)

        # k > 4: all in top
        hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=4))
        self.assertEqual(hr, 1.0)
        hr = self._eval_tensor(hit_rate(y_true, y_pred, num_negs_per_pos, k=5))
        self.assertEqual(hr, 1.0)

    def _dcg_index(self, index):
        return math.log(2) / math.log(index + 2)

    def test_dcg(self):
        # last elem in each row is expected label (should be max in pred)
        y_true = np.array([[0, 0, 0, 1],
                           [0, 0, 0, 1]])
        y_pred = np.array([[0.1, 0.2, 0.9, 0.5],
                           [0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
        num_negs_per_pos = 3
        batch_size = 2

        # compute individual DCG. In first row label is ranked 1 (0-indexed) and in row 2, 3
        dcg0 = self._dcg_index(1)
        dcg1 = self._dcg_index(3)

        # for top 0, top 1: no hit: dcg=0
        avg_dcg = self._eval_tensor(dcg(y_true, y_pred, batch_size, num_negs_per_pos, k=0))
        self.assertEqual(avg_dcg, 0.0)
        avg_dcg = self._eval_tensor(dcg(y_true, y_pred, batch_size, num_negs_per_pos, k=1))
        self.assertEqual(avg_dcg, 0.0)

        # top 2, top 3: first row is hit, second is not
        expected_avg = (dcg0 + 0.0) / 2.
        avg_dcg = self._eval_tensor(dcg(y_true, y_pred, batch_size, num_negs_per_pos, k=2))
        self.assertAlmostEqual(avg_dcg, expected_avg)
        avg_dcg = self._eval_tensor(dcg(y_true, y_pred, batch_size, num_negs_per_pos, k=3))
        self.assertAlmostEqual(avg_dcg, expected_avg)

        # k = 4: all in top
        expected_avg = (dcg0 + dcg1) / 2.
        avg_dcg = self._eval_tensor(dcg(y_true, y_pred, batch_size, num_negs_per_pos, k=4))
        self.assertAlmostEqual(avg_dcg, expected_avg)
