
import argparse
import numpy as np
from tensorrtserver.api import ProtocolType, InferContext
import util.movielens_utils as ml


COL_USER_ID = 'userId'
COL_MOVIE_ID = 'movieId'
COL_MOVIE_TITLE = 'movieTitle'

NUM_ITEMS_PREDICT = 1000
K = 10


def call_predict(model_name, model_version, url, data_dir, dataset_name, verbose):

    print("----------------\n"
          "-   MOVIEREC   -\n"
          "----------------\n"
          "\nLoading dataset (this might take a while but it's only required once on startup)")

    ratings_df = ml.load_ratings_data(data_dir, dataset_name, COL_USER_ID, COL_MOVIE_ID)
    movies_df = ml.load_movies_data(data_dir, dataset_name, COL_MOVIE_ID, COL_MOVIE_TITLE)

    print("Done.\n\nMovierec: provide movie recommendations to users based on their watch history.\n"
          "This script queries a model running in TensorRT Inference Server and prints recommendations.\n"
          "The inference request sent to the model is done with a random set of movies, and the top {} "
          "are selected. Recommendations will be different every attempt.\n".format(K))
    while True:
        user_id = input("Enter movielens user ID (integer between 0 and {}) or 'exit' to exit... ".format(ml.NUM_USERS[dataset_name] - 1))
        if user_id.strip() == 'exit':
            return
        try:
            user_id = int(user_id)
            assert user_id >= 0 and user_id < ml.NUM_USERS[dataset_name]
        except:
            print('Invalid user ID. Must be an integer between 0 and {}'.format(ml.NUM_USERS[dataset_name] - 1))
            continue

        # Create the inference context for the model.
        protocol = ProtocolType.from_str('http')
        ctx = InferContext(url, protocol, model_name, model_version, verbose=verbose)

        # Create the data batch for the two input tensors, with just one user and NUM_ITEMS_PREDICT random items
        # simple demo, there might be repetitions or movies already watched
        users = [np.array([user_id], dtype='int32')] * NUM_ITEMS_PREDICT
        items = [np.random.randint(ml.NUM_ITEMS[dataset_name], size=1, dtype='int32') for _ in range(NUM_ITEMS_PREDICT)]

        # Send inference request to the inference server.
        result = ctx.run({'user_input': users, 'item_input': items},
                         {'output/Sigmoid': InferContext.ResultFormat.RAW},
                         NUM_ITEMS_PREDICT)
        # Sort results to get top K
        output = np.array( result['output/Sigmoid']).reshape((-1,))
        top_k_idx = np.argsort(output)[-K:][::-1]
        top_k_items = np.array(items)[top_k_idx].reshape((-1,))

        # Print results
        sample_movies = ratings_df[ratings_df[COL_USER_ID] == user_id][:20]
        sample_movies = sample_movies[COL_MOVIE_ID].map(movies_df.set_index(COL_MOVIE_ID)[COL_MOVIE_TITLE])
        print("\nSome movies watched by user {} are:\n   -{}".format(user_id, '\n   -'.join(sample_movies.values)))

        print("\n Recommended movies:\n")
        if verbose:
            print('rank  score  title')
        for i, item in enumerate(top_k_items):
            print(' {:2d}. {}{}'.format(i+1,
                                        ' {:.2f}  '.format(output[top_k_idx[i]]) if verbose else '',
                                        movies_df[movies_df[COL_MOVIE_ID] == item][COL_MOVIE_TITLE].values[0]))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get movie recommendations for a user calling'
                                                 'a running TensorRT inference server')
    parser.add_argument('-d', '--data-dir', type=str, default='data/', help='Movielens dataset directory.')
    parser.add_argument('-n', '--dataset-name', type=str, required=True, help='Movielens dataset name.')
    parser.add_argument('-m', '--model-name', type=str, default='movierec', help='Model name.')
    parser.add_argument('-vr', '--model-version', type=int, default=1, help='Model version.')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False, help='Generate verbose output.')

    args = parser.parse_args()

    call_predict(args.model_name, args.model_version, args.url, args.data_dir, args.dataset_name, args.verbose)
