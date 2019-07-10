""" Recommender model and utility methods to train, predict and evaluate """

import functools
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import concatenate, Dense, Embedding, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.regularizers import l2

DEFAULT_PARAMS = {
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
    "k": 4
}

ADAM = "adam"
SGD = "sgd"
OPTIMIZERS = [ADAM, SGD]


class MovierecModel(object):
    """
    Movie Recommendation Model
    """

    def __init__(self, params=DEFAULT_PARAMS):
        """
        Create a movie recommendation model.
        Parameters
        ----------
        Parameters
        ----------
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

        self._k = params.get("k", self._num_negs_per_pos + 1)  # optional
        if self._k > (self._num_negs_per_pos + 1):
            raise ValueError("'k' must be lower than (num_negs_per_pos + 1). Found: k={}, "
                             "num_negs_per_pos={}".format(self._k, self._num_negs_per_pos))

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
        mlp_layer = concatenate([user_embedding(user_input), item_embedding(item_input)])

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
            kernel_initializer="lecun_uniform", name="output"
        )
        output = pred_layer(mlp_layer)

        # Create Model
        model = Model([user_input, item_input], output)
        return model

    def compile_model(self):
        # Set optimizer
        if self._optimizer is ADAM:
            optimizer = Adam(self._lr, self._beta_1, self._beta_2)
        elif self._optimizer is SGD:
            optimizer = SGD(self._lr)
        else:
            raise NotImplementedError("Optimizer {} is not implemented.".format(self._optimizer))

        # Set metrics
        hit_rate_fn = functools.partial(hit_rate, num_negs_per_pos=self._num_negs_per_pos, k=self._k)
        hit_rate_fn.__name__ = 'HR'
        dcg_fn = functools.partial(dcg, batch_size=self._batch_size, num_negs_per_pos=self._num_negs_per_pos, k=self._k)
        dcg_fn.__name__ = 'DCG'

        # Compile model
        self.model.compile(optimizer=optimizer,
                           loss="binary_crossentropy",
                           metrics=[hit_rate_fn, dcg_fn])

    def log_summary(self):
        self.model.summary(print_fn=logging.info)

    def save(self, output_model_file="models/movirec.h5"):
        # Save model
        try:
            os.makedirs(os.path.dirname(output_model_file))
        except FileExistsError:
            # directory already exists
            pass
        self.model.save(output_model_file)
        logging.info('Keras model saved to {}'.format(output_model_file))


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
    y_pred_per_user = K.reshape(y_pred, (-1, num_negs_per_pos + 1))
    labels_per_user = K.math_ops.argmax(K.reshape(y_true, (-1, num_negs_per_pos + 1)), axis=-1)
    return K.mean(K.in_top_k(y_pred_per_user, labels_per_user, k), axis=-1)


def dcg(y_true, y_pred, batch_size, num_negs_per_pos, k):
    """
    Compute DCG (Discounted Cummulative Gain) considering
    only the top 'k' items in the rank.

    Parameters
    ----------
    y_true : `tf.Tensor` (or `np.array`)
        True labels. For every (`num_negs_per_pos` + 1) items, there should be only one positive class (+1)
        and the rest are negative (0).
    y_pred : `tf.Tensor` (or `np.array`)
        Predicted logits.
    batch_size : int
        Size of the batch.
    num_negs_per_pos : int
        Number of negative examples for each positive one (for the same user).
    k : int
        Number of top elements to consider for the metric computation.

    Returns
    -------
    hit rate: `tf.Tensor`
        A single value tensor with the average DCG on top k for the batch.
    """
    y_pred_per_user = K.reshape(y_pred, (-1, num_negs_per_pos + 1))

    labels_per_user = K.reshape(y_true, (-1, num_negs_per_pos + 1))
    labels_per_user = K.math_ops.argmax(labels_per_user, axis=-1, output_type=K.dtypes_module.int32)

    values, indices = K.nn.top_k(y_pred_per_user, k, sorted=True)

    # get the position of the label in the ranked list (if not in top K, position not returned)
    pos_label = K.array_ops.where(K.equal(indices, K.reshape(labels_per_user, (-1, 1))))[:, -1]

    dcg_batch = K.math_ops.log(2.) / K.math_ops.log(K.cast(pos_label, K.dtypes_module.float32) + 2)

    # Compute mean.
    # If label not in top K, count as 0.0, and since dcg_batch does not include 0s: mean = sum(dcg_batch) / bach_size
    return K.sum(dcg_batch) / batch_size
