""" Recommender model and utility methods to train, predict and evaluate """

import functools
import logging
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import concatenate, Dense, Embedding, Input, Flatten, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.regularizers import l2


DEFAULT_PARAMS = {  # Just some toy params to test the code

    # model:
    "num_users": 5,
    "num_items": 10,
    "layers_sizes": [5, 4],
    "layers_l2reg": [0.01, 0.01],

    # training:
    "optimizer": "adam",
    "lr": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,

    "batch_size": 35,
    "num_negs_per_pos": 4,
    "batch_size_eval": 35,
    "num_negs_per_pos_eval": 4,
    "k": 4
}

ADAM = "adam"
SGD = "sgd"
OPTIMIZERS = [ADAM, SGD]

HIT_RATE = "hr"
DCG = "dcg"

OUTPUT_PRED = "output"
OUTPUT_RANK = "rank"

METRIC_VAL_DCG = "val_{}_{}".format(OUTPUT_PRED, DCG)


class MovierecModel(object):
    """
    Movie Recommendation Model
    """

    def __init__(self, params=DEFAULT_PARAMS, output_model_file="models/movirec.h5", ):
        """
        Create a movie recommendation model.

        Parameters
        ----------
        output_model_file : str or `os.path`
            Output file to save the Keras model (HDF5 format).
        params : dict of param names (str) to values (any type)
           Dictionary of model hyper parameters. Default: `DEFAULT_PARAMS`


        """
        # Get params and perform some basic verifications
        self._num_users = params["num_users"]
        self._num_items = params["num_items"]
        self._layers_sizes = params["layers_sizes"]
        self._layers_l2reg = params["layers_l2reg"]
        if len(self._layers_sizes) != len(self._layers_l2reg):
            raise ValueError("'layers_sizes' length = {}, 'layers_l2reg' length = {}, but must be equal."
                             .format(len(self._layers_sizes), len(self._layers_l2reg)))
        self._num_layers = len(self._layers_sizes)

        # params to compile
        self._optimizer = params["optimizer"]
        if self._optimizer not in OPTIMIZERS:
            raise NotImplementedError("Optimizer {} is not implemented.".format(params["optimizer"]))
        self._lr = params["lr"]
        self._beta_1 = params.get("beta_1", 0.9) # optional, for Adam optimizer, default from paper
        self._beta_2 = params.get("beta_2", 0.999) # optional, for Adam optimizer, default from paper
        self._batch_size = params["batch_size"]
        self._num_negs_per_pos = params["num_negs_per_pos"]
        if self._num_negs_per_pos <= 0:
            raise ValueError("num_negs_per_pos must be > 0, found {}".format(self._num_negs_per_pos))

        if self._batch_size % (self._num_negs_per_pos + 1):
            raise ValueError("Batch size must be divisible by (num_negs_per_pos + 1). Found: batch_size={}, "
                             "num_negs_per_pos={}".format(self._batch_size, self._num_negs_per_pos))

        self._batch_size_eval = params["batch_size_eval"]
        self._num_negs_per_pos_eval = params["num_negs_per_pos_eval"]
        if self._num_negs_per_pos_eval <= 0:
            raise ValueError("num_negs_per_pos_eval must be > 0, found {}".format(self._num_negs_per_pos_eval))

        if self._batch_size_eval % (self._num_negs_per_pos_eval + 1):
            raise ValueError("Batch size (eval) must be divisible by (num_negs_per_pos_eval + 1). Found: "
                             "batch_size_eval={}, num_negs_per_pos_eval={}".format(self._batch_size_eval,
                                                                                   self._num_negs_per_pos_eval))

        self._k = params.get("k", self._num_negs_per_pos + 1)  # optional
        if self._k > (self._num_negs_per_pos + 1):
            raise ValueError("'k' must be lower than (num_negs_per_pos + 1) and lower than (num_negs_per_pos_eval + 1)."
                             "Found: k={}, num_negs_per_pos={}, num_negs_per_pos_eval={}"
                             .format(self._k, self._num_negs_per_pos, self._num_negs_per_pos_eval))

        # Create output dir and get file names
        model_dir = os.path.dirname(output_model_file)
        try:
            os.makedirs(model_dir)
        except FileExistsError:
            # directory already exists
            pass
        self._output_model_file = output_model_file
        self._output_model_checkpoints = os.path.join(model_dir, "checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5")

        # Build model and compile
        self.model = self.build_mlp_model()
        self.compile_model()

    def build_mlp_model(self):
        """
        Build a MLP (Multi Layer Perceptron) model with the following architecture:

        Input: 2 matrices of size (batch, num_users), (batch, num_items)

        First Layer:  [ User Embedding ][ Item Embedding ]
        N times:           [ Hidden Dense Layer ]
        Last Layer:         [ Prediction Layer ]

        Output:  vector of size (batch, 1)

        Returns
        -------
        model: `Model`
            Keras MLP model.
        """

        # Inputs
        user_input = Input(shape=(1,), dtype="int32", name="user_input")
        item_input = Input(shape=(1,), dtype="int32", name="item_input")

        # First layer is the concatenation of embeddings for users and items
        # (size of each is about half of layers_sizes[0])
        user_layer_size = self._layers_sizes[0] // 2
        item_layer_size = self._layers_sizes[0] - user_layer_size  # in case layers_sizes[0] % 2 != 0
        user_embedding = Embedding(
            input_dim=self._num_users, output_dim=user_layer_size, input_length=1,
            embeddings_initializer="glorot_uniform", embeddings_regularizer=l2(self._layers_l2reg[0]),
            name="user_embedding"
        )
        item_embedding = Embedding(
            input_dim=self._num_items, output_dim=item_layer_size, input_length=1,
            embeddings_initializer="glorot_uniform", embeddings_regularizer=l2(self._layers_l2reg[0]),
            name="item_embedding"
        )
        mlp_layer = concatenate([Flatten()(user_embedding(user_input)),
                                 Flatten()(item_embedding(item_input))])

        # Hidden layers
        for layer_i in range(1, self._num_layers):
            hidden = Dense(
                units=self._layers_sizes[layer_i], activation="relu",
                kernel_initializer="glorot_uniform", kernel_regularizer=l2(self._layers_l2reg[layer_i]),
                name="hidden_{}".format(layer_i)
            )
            mlp_layer = hidden(mlp_layer)

        # Prediction layer
        pred_layer = Dense(
            units=1, activation="sigmoid",
            kernel_initializer="lecun_uniform", name=OUTPUT_PRED
        )
        output_pred = pred_layer(mlp_layer)

        rank_layer = RankLayer(self._num_negs_per_pos, self._num_negs_per_pos_eval, name=OUTPUT_RANK)
        rank = rank_layer(output_pred)

        # Create Model
        model = Model([user_input, item_input], [output_pred, rank])
        return model

    def compile_model(self):
        # Set optimizer
        if self._optimizer is ADAM:
            optimizer = Adam(self._lr, self._beta_1, self._beta_2)
        elif self._optimizer is SGD:
            optimizer = SGD(self._lr)
        else:
            raise NotImplementedError("Optimizer {} is not implemented.".format(self._optimizer))

        # Create metrics
        # TODO: consider _num_negs_per_pos_eval for validation!
        hit_rate_fn = functools.partial(hit_rate, num_negs_per_pos=self._num_negs_per_pos, k=self._k)
        hit_rate_fn.__name__ = HIT_RATE
        dcg_fn = functools.partial(discounted_cumulative_gain, num_negs_per_pos=self._num_negs_per_pos, k=self._k)
        dcg_fn.__name__ = DCG

        # Compile model
        self.model.compile(optimizer=optimizer,
                           loss={OUTPUT_PRED: "binary_crossentropy"},
                           metrics={OUTPUT_PRED: [hit_rate_fn, dcg_fn]})

    def log_summary(self):
        self.model.summary(print_fn=logging.info)

    def save(self):
        # Save model
        self.model.save(self._output_model_file)
        logging.info('Keras model saved to {}'.format(self._output_model_file))

    def fit_generator(self, train_data_generator, validation_data_generator, epochs):
        """
        Call keras 'fit_generator' on the model with early stopping and checkpoint callbacks.

        Parameters
        ----------
        train_data_generator : A generator or a `keras.utils.Sequence`
            Generator of training data.
        validation_data_generator : A generator or a `keras.utils.Sequence`
            Generator of validation data.
        epochs : int
            Number of epochs.

        Returns
        -------
            Output of model.fit_generator(...) (`History` object)
        """
        # Callbacks
        callbacks = [
            EarlyStopping(monitor=METRIC_VAL_DCG, mode='max', restore_best_weights=True),
            ModelCheckpoint(self._output_model_checkpoints, monitor=METRIC_VAL_DCG, save_best_only=True,
                            save_weights_only=False, mode='max')
        ]
        return self.model.fit_generator(generator=train_data_generator,
                                        validation_data=validation_data_generator,
                                        epochs=epochs,
                                        callbacks=callbacks)


class RankLayer(Layer):

    def __init__(self, num_negs_per_pos_train, num_negs_per_pos_eval, name):
        super(RankLayer, self).__init__(name=name)
        self.num_negs_per_pos_train = num_negs_per_pos_train
        self.num_negs_per_pos_eval = num_negs_per_pos_eval
        self._uses_learning_phase = True

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        inputs = K.ops.convert_to_tensor(inputs)

        num_negs_per_pos = K.in_train_phase(self.num_negs_per_pos_train, self.num_negs_per_pos_eval)

        # Reshape and get ranked indices per user
        y_pred_per_user = K.reshape(inputs, (-1, num_negs_per_pos + 1))
        _, indices = K.nn.top_k(y_pred_per_user, K.shape(y_pred_per_user)[1], sorted=True)
        return indices


def hit_rate(y_true, y_pred, num_negs_per_pos, k):
    """
    Compute HR (Hit Rate) in batch considering only the top 'k' items in the rank.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    y_pred : `tf.Tensor` (or `np.array`)
        Predicted logits.
    num_negs_per_pos : int
        Number of negative examples for each positive one (for the same user).
    k : int
        Number of top elements to consider for the metric computation.

    Returns
    -------
    hit rate: `tf.Tensor`
        A single value tensor with the hit rate for the batch.
    """
    return _dcg_hr(y_true, y_pred, num_negs_per_pos, k)[1]


def discounted_cumulative_gain(y_true, y_pred, num_negs_per_pos, k):
    """
    Compute DCG (Discounted Cumulative Gain) considering only the top 'k' items in the rank.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    y_pred : `tf.Tensor` (or `np.array`)
        Predicted logits.
    num_negs_per_pos : int
        Number of negative examples for each positive one (for the same user).
    k : int
        Number of top elements to consider for the metric computation.

    Returns
    -------
    hit rate: `tf.Tensor`
        A single value tensor with the average Hit Rate on the top k for the batch.
    """
    return _dcg_hr(y_true, y_pred, num_negs_per_pos, k)[0]


def _dcg_hr(y_true, y_pred, num_negs_per_pos, k):
    """
    Compute DCG (Discounted Cumulative Gain) and Hit Rate considering only the top 'k' items in the rank.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    y_pred : `tf.Tensor` (or `np.array`)
        Predicted logits.
    num_negs_per_pos : int
        Number of negative examples for each positive one (for the same user).
    k : int
        Number of top elements to consider for the metric computation.

    Returns
    -------
    hit rate: Tuple of 2 `tf.Tensor`
        dcg: A single value tensor with the average DCG on the top k for the batch.
        hr: A single value tensor with the average Hit Rate on the top k for the batch.
    """
    # Get predictions and labels per user
    y_pred_per_user = K.reshape(y_pred, (-1, num_negs_per_pos + 1))
    labels_per_user = K.reshape(y_true, (-1, num_negs_per_pos + 1))
    labels_per_user = K.math_ops.argmax(labels_per_user, axis=-1, output_type=K.dtypes_module.int32)

    # get rank indices per user
    _, indices = K.nn.top_k(y_pred_per_user, K.shape(y_pred_per_user)[1], sorted=True)

    # get the position of the expecgtedvlabel in the ranked list
    pos_label = K.array_ops.where(K.equal(indices, K.reshape(labels_per_user, (-1, 1))))[:, -1]

    # compute dcg for each item, but make 0.0 the entries where position is > k (only consider top k)
    dcg_per_user = K.math_ops.log(2.) / K.math_ops.log(K.cast(pos_label, K.dtypes_module.float32) + 2)
    hits_per_user = K.cast(K.less(pos_label, k), K.dtypes_module.float32)
    dcg_per_user *= hits_per_user

    # return mean dcg and hit rate
    dcg = K.mean(dcg_per_user, axis=-1)
    hr = K.mean(hits_per_user, axis=-1)
    return dcg, hr
