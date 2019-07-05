"""" Train and evaluate models. """

import data_pipeline
import logging
from model import build_mlp_model, compile_model
import os
from tensorflow.python.keras import backend as K

DEFAULT_PARAMS = {
    # model:
    # "num_users": obtained from data generator,
    # "num_items": obtained from data generator,
    "layers_sizes": [10, 10],
    "layers_l2reg": [0.01, 0.01],

    # training:
    "optimizer": "adam",
    "lr": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,

    "batch_size": 1200,
    "eval_batch_size": 300,
    "epochs": 1, #1000
    "num_negs_per_pos": 4
}


def train(dataset_name, data_dir, output_model_file, params=DEFAULT_PARAMS):
    """
    Create dataset train and validation generators, create and compile model and train the model.
    Parameters
    ----------
    dataset_name : str
            Movielens dataset name. Must be one of MOVIELENS_DATASET_NAMES.
    data_dir : str or os.path
        Dataset directory to read ratings data from. The file to read from the directory will be:
        data_dir/dataset_name/data_pipeline.RATINGS_FILE_NAME[dataset_name].
    output_model_file : str or `os.path`
        Output file to save the Keras model (HDF5 format).
    params : dict of param names (str) to values (any type)
       Dictionary of model hyper parameters. Default: `DEFAULT_PARAMS`

    """

    # TODO: cache data?
    train_df, validation_df, test_df = data_pipeline.load_ratings_train_test_sets(dataset_name, data_dir)

    # Train and validation data generators.
    train_data_generator = data_pipeline.MovieLensDataGenerator(
        dataset_name,
        train_df,
        params["batch_size"],
        params["num_negs_per_pos"],
        extra_data_df=None,  # Don't use any validation/test info.
                             # Some negatives in train might be positives in val/test.
        shuffle=True)
    validation_data_generator = data_pipeline.MovieLensDataGenerator(
        dataset_name,
        validation_df,
        params["eval_batch_size"],
        params["num_negs_per_pos"],
        extra_data_df=train_df,  # Use train to avoid positives from train in validation.
        shuffle=False)

    # update params:
    params["num_users"] = train_data_generator.num_users
    params["num_items"] = train_data_generator.num_items

    # Create and train the model
    model = build_mlp_model(params)
    compile_model(model, params)

    # set learning phase to 'train'
    K.set_learning_phase(1)
    model.fit_generator(generator=train_data_generator,
                        epochs=params["epochs"],
                        validation_data=validation_data_generator)

    # Save model
    try:
        os.makedirs( os.path.dirname(output_model_file))
    except FileExistsError:
        # directory already exists
        pass
    model.save(output_model_file)
    logging.info('Keras model saved to {}'.format(output_model_file))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train a Keras model.')
    parser.add_argument('-n', '--dataset-name', type=str, required=True,
                        help='Movielens dataset name.')
    parser.add_argument('-d', '--data-dir', type=str, default='data',
                        help='Dataset directory to read ratings data from')
    parser.add_argument('-o', '--output-model-file', type=str, default='models/movierec_model.h5',
                        help='Output file to save the Keras model')
    # TODO `params` arg
    args = parser.parse_args()

    train(args.dataset_name, args.data_dir, args.output_model_file)
