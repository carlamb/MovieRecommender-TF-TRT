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

    "batch_size": 6,
    "num_negs_per_pos": 2,
    "batch_size_eval": 12,
    "num_negs_per_pos_eval": 5,
    "k": 3
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

    def __init__(self, params=DEFAULT_PARAMS, output_model_file="models/movirec.h5", verbose=1):
        """
        Create a movie recommendation model.

        Parameters
        ----------
        params : dict of param names (str) to values (any type)
           Dictionary of model hyper parameters. Default: `DEFAULT_PARAMS`
        output_model_file : str or `os.path`
            Output file to save the Keras model (HDF5 format).
        verbose : int
            Verbosity mode.

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
        self.verbose = verbose

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
        hit_rate_fn = functools.partial(hit_rate, k=self._k, pred_rank_idx=self.get_pred_rank())
        hit_rate_fn.__name__ = HIT_RATE
        dcg_fn = functools.partial(discounted_cumulative_gain, k=self._k, pred_rank_idx=self.get_pred_rank())
        dcg_fn.__name__ = DCG

        # Compile model
        self.model.compile(optimizer=optimizer,
                           loss={OUTPUT_PRED: "binary_crossentropy"},
                           metrics={OUTPUT_PRED: [hit_rate_fn, dcg_fn]})

    def get_pred_rank(self):
        return self.model.get_layer(OUTPUT_RANK).output

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
            EarlyStopping(monitor=METRIC_VAL_DCG, mode='max', restore_best_weights=True, patience=3,
                          verbose=self.verbose),
            ModelCheckpoint(self._output_model_checkpoints, monitor=METRIC_VAL_DCG, save_best_only=True,
                            save_weights_only=False, mode='max', verbose=self.verbose)
        ]
        return self.model.fit_generator(generator=train_data_generator,
                                        validation_data=validation_data_generator,
                                        epochs=epochs,
                                        callbacks=callbacks,
                                        verbose=self.verbose)


class RankLayer(Layer):

    def __init__(self, num_negs_per_pos_train, num_negs_per_pos_eval, name, **kwargs):
        super(RankLayer, self).__init__(name=name, **kwargs)
        self.num_negs_per_pos_train = num_negs_per_pos_train
        self.num_negs_per_pos_eval = num_negs_per_pos_eval
        self._uses_learning_phase = True

    def call(self, inputs):
        inputs = K.ops.convert_to_tensor(inputs)

        num_negs_per_pos = K.in_train_phase(self.num_negs_per_pos_train, self.num_negs_per_pos_eval)

        # Reshape and get ranked indices per user
        y_pred_per_user = K.reshape(inputs, (-1, num_negs_per_pos + 1))
        _, indices = K.nn.top_k(y_pred_per_user, K.shape(y_pred_per_user)[1], sorted=True)
        return indices

    def get_config(self):
        config = {'num_negs_per_pos_train': self.num_negs_per_pos_train,
                  'num_negs_per_pos_eval': self.num_negs_per_pos_eval}
        base_config = super(RankLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def hit_rate(y_true, y_pred, k, pred_rank_idx):
    """
    Compute HR (Hit Rate) in batch considering only the top 'k' items in the rank.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    y_pred : `tf.Tensor` (or `np.array`)
        Predicted logits. Ignored argument that will be passed by keras metrics, but this method will only
        use pred_rank_idx.
    k : int
        Number of top elements to consider for the metric computation.
    pred_rank_idx : `tf.Tensor`(or `np.array`) of integers. Shape: (users per batch, num_negs_per_pos + 1)
        Tensor representing a ranking. Each row represents a single user and contains (num_negs_per_pos + 1) elements
        with the ranked indexes of the items in the row.

    Returns
    -------
    hit rate: `tf.Tensor`
        A single value tensor with the hit rate for the batch.
    """
    hits_per_user, _ = _get_hits_per_user(y_true, pred_rank_idx, k)
    return K.mean(hits_per_user, axis=-1)


def discounted_cumulative_gain(y_true, y_pred, k, pred_rank_idx):
    """
    Compute DCG (Discounted Cumulative Gain) considering only the top 'k' items in the rank.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    y_pred : `tf.Tensor` (or `np.array`)
        Predicted logits. Ignored argument that will be passed by keras metrics, but this method will only
        use pred_rank_idx.
    k : int
        Number of top elements to consider for the metric computation.
    pred_rank_idx : `tf.Tensor`(or `np.array`) of integers. Shape: (users per batch, num_negs_per_pos + 1)
        Tensor representing a ranking. Each row represents a single user and contains (num_negs_per_pos + 1) elements
        with the ranked indexes of the items in the row.

    Returns
    -------
    discounted cumulative gain: `tf.Tensor`
        A single value tensor with the average Discounted Cumulative Gain on the top k for the batch.
    """
    hits_per_user, idx_label_in_pred_rank = _get_hits_per_user(y_true, pred_rank_idx, k)

    # compute dcg for each item, but make 0.0 the entries where position is > k (only consider top k)
    dcg_per_user = K.math_ops.log(2.) / K.math_ops.log(K.cast(idx_label_in_pred_rank, "float32") + 2)
    dcg_per_user *= hits_per_user

    return K.mean(dcg_per_user, axis=-1)


def _get_hits_per_user(y_true, pred_rank_idx, k):
    """
    Compute the position of the label in the predicted ranking and whether is a hit on top k or not.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    pred_rank_idx : `tf.Tensor`(or `np.array`) of integers. Shape: (users per batch, num_negs_per_pos + 1)
        Tensor representing a ranking. Each row represents a single user and contains (num_negs_per_pos + 1) elements
        with the ranked indexes of the items in the row.
    k : int
        Number of top elements to consider for the metric computation.

    Returns
    -------
    Tuple: (hits_per_user, idx_label_in_pred_rank)
        hits_per_user : `tf.Tensor` with shape (users per batch, )
            Tensor of floats where elements are 1.0 if there is a hit (label is in top k) for that user, or
            0.0 otherwise.
        idx_label_in_pred_rank : `tf.Tensor` with shape (users per batch, )
            Tensor of integers with the index of the label in the predicted rank.
    """

    # Get the index of the positive label per user.
    # Assume that every user has num_neg_per_pos negatives (zeros) and one positive (1).
    idx_label_per_user = K.reshape(y_true, K.shape(pred_rank_idx))
    idx_label_per_user = K.math_ops.argmax(idx_label_per_user, axis=-1, output_type="int32")

    # get the position of the expected label in the ranked list and compute whether is a hit in top k
    idx_label_in_pred_rank = K.array_ops.where(K.equal(pred_rank_idx, K.reshape(idx_label_per_user, (-1, 1))))[:, -1]

    # determine whether the label is in top k of ranking or not
    hits_per_user = K.cast(K.less(idx_label_in_pred_rank, k), "float32")
    return hits_per_user, idx_label_in_pred_rank

