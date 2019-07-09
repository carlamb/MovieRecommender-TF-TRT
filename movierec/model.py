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

    "num_negs_per_pos": 4,
    "k": 4
}


def build_mlp_model(params=DEFAULT_PARAMS):
    """
    Build a MLP (Multi Layer Perceptron) model with the following architecture:

    Input: 2 matrices of size (batch, num_users), (batch, num_items)

    First Layer:  [ User Embedding ][ Item Embedding ]
    N times:           [ Hidden Dense Layer ]
    Last Layer:         [ Prediction Layer ]

    Output:  vector of size (batch, 1)

    Parameters
    ----------
    params : dict of param names (str) to values (any type)
       Dictionary of model hyper parameters. Default: `DEFAULT_PARAMS`

    Returns
    -------
    model: `Model`
        Keras MLP model.
    """
    num_users = params["num_users"]
    num_items = params["num_items"]
    layers_sizes = params["layers_sizes"]
    layers_l2reg = params["layers_l2reg"]
    if len(layers_sizes) != len(layers_l2reg):
        raise ValueError("'layers_sizes' length = {}, 'layers_l2reg' length = {}, but must be equal."
                         .format(len(layers_sizes), len(layers_l2reg)))

    # Inputs
    user_input = Input(shape=(1,), dtype="int32", name="user_input")
    item_input = Input(shape=(1,), dtype="int32", name="item_input")

    # First layer is the concatenation of embeddings for users and items (size of each is about half of layers_sizes[0])
    user_layer_size = layers_sizes[0] // 2
    item_layer_size = layers_sizes[0] - user_layer_size  # in case layers_sizes[0] % 2 != 0
    user_embedding = Embedding(
        input_dim=num_users, output_dim=user_layer_size, input_length=1,
        embeddings_initializer="glorot_uniform", embeddings_regularizer=l2(layers_l2reg[0]), name="user_embedding"
    )
    item_embedding = Embedding(
        input_dim=num_items, output_dim=item_layer_size, input_length=1,
        embeddings_initializer="glorot_uniform", embeddings_regularizer=l2(layers_l2reg[0]), name="item_embedding"
    )
    mlp_layer = concatenate([user_embedding(user_input), item_embedding(item_input)])

    # Hidden layers
    num_layers = len(layers_sizes)
    for layer_i in range(1, num_layers):
        hidden = Dense(
            units=layers_sizes[layer_i], activation="relu",
            kernel_initializer="glorot_uniform", kernel_regularizer=l2(layers_l2reg[layer_i]),
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
    model.summary()

    return model


def compile_model(model, params=DEFAULT_PARAMS):
    # Set optimizer
    if params["optimizer"] == "adam":
        optimizer = Adam(params["lr"], params["beta_1"], params["beta_2"])
    elif params["optimizer"] == "sgd":
        optimizer = SGD(params["lr"])
    else:
        raise NotImplementedError("Optimizer {} is not implemented.".format(params["optimizer"]))

    # Set metrics
    hit_rate_fn = functools.partial(hit_rate, num_negs_per_pos=params["num_negs_per_pos"], k=params["k"])
    hit_rate_fn.__name__ = 'HR'

    # Compile model
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=[hit_rate_fn])


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

