from movierec.model import build_mlp_model, compile_model

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
    "beta_2": 0.999
}


class TestModel(TestCase):

    def test_wrong_layers(self):
        params = TEST_PARAMS.copy()
        # add one more layer to "layers_sizes", but not to "layers_l2reg" to generate error
        params["layers_sizes"].append(2)
        self.assertRaises(ValueError, build_mlp_model, params)

    def test_missing_param(self):
        params = TEST_PARAMS.copy()
        del params["num_users"]
        self.assertRaises(KeyError, build_mlp_model, params)

    def test_build_mlp_model(self):
        model = build_mlp_model(TEST_PARAMS)

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
        self.assertRaises(NotImplementedError, compile_model, "model", {"optimizer": "Other"})
