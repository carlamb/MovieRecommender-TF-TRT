"""" Train and evaluate models. """

import data_pipeline
import logging
from model import MovierecModel
import os

DEFAULT_PARAMS = {
    # model:
    # "num_users": obtained from data generator,
    # "num_items": obtained from data generator,
    "layers_sizes": [64, 32, 16, 8],
    "layers_l2reg": [0, 0, 0, 0],

    # training:
    "optimizer": "adam",
    "lr": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,

    "batch_size": 100,
    "batch_size_eval": 200,
    "num_negs_per_pos": 9,
    "num_negs_per_pos_eval": 99,
    "k": 5,
    "epochs": 20,
}


def train(model_name, dataset_name, data_dir, output_dir, params=DEFAULT_PARAMS, verbose=1):
    """
    Create dataset train and validation generators, create and compile model and train the model.
    Parameters
    ----------
    model_name : str
        Name of the model to train (used in model object and model files to save).
    dataset_name : str
        Movielens dataset name. Must be one of MOVIELENS_DATASET_NAMES.
    data_dir : str or os.path
        Dataset directory to read ratings data from. The file to read from the directory will be:
        data_dir/dataset_name/data_pipeline.RATINGS_FILE_NAME[dataset_name].
    output_dir : str or `os.path`
        Output file directory to save model files.
    params : dict of param names (str) to values (any type)
       Dictionary of model hyper parameters. Default: `DEFAULT_PARAMS`
    verbose : int
        Verbosity mode.

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
        params["batch_size_eval"],
        params["num_negs_per_pos_eval"],
        extra_data_df=train_df,  # Use train to avoid positives from train in validation.
        shuffle=False)

    # update users and items (obtained from data) and create model:
    params["num_users"] = train_data_generator.num_users
    params["num_items"] = train_data_generator.num_items

    movierec_model = MovierecModel(params, model_name, output_dir, verbose)
    movierec_model.log_summary()

    movierec_model.fit_generator(train_data_generator, validation_data_generator, params["epochs"])

    movierec_model.save()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train a Keras model.')
    parser.add_argument('-m', '--model-name', type=str, required=True,
                        help='Model name (to save output files).')
    parser.add_argument('-n', '--dataset-name', type=str, required=True,
                        help='Movielens dataset name.')
    parser.add_argument('-d', '--data-dir', type=str, default='data/',
                        help='Dataset directory to read ratings data from')
    parser.add_argument('-o', '--output-dir', type=str, default='models',
                        help='Output dir to save model files.')
    parser.add_argument('-l', '--log-level', type=str, default='INFO',
                        help='Log level (default: INFO).')
    # TODO Allow `params` as argument

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.log_level))
    logging.info("Starting training with params: {}".format(DEFAULT_PARAMS))
    train(args.model_name, args.dataset_name, args.data_dir, args.output_dir, DEFAULT_PARAMS, logging.getLogger().level)
