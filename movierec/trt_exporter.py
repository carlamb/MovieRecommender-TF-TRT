""" Export models. """

import logging
from model import MovierecModel
import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_io

# For versions older than 1.14, use graph_utils from 1.14 (locally downloaded)
tf_version = tf.__version__.split('.')
if tf_version[0] == '1' and int(tf_version[1]) < 14:
    logging.warn("Importing graph_util from local copy because tensorflow version {} is lower than 1.14"
                 .format(tf.__version__))
    from util import tf_graph_util
else:
    import tensorflow.compat.v1.graph_util as tf_graph_util


def export_keras_model_to_trt(input_dir, model_name, output_dir):
    """
    Export a saved keras model to a TensorRT-compatible model. Steps: load keras model from file,
    freeze and optimize graph for inference, save in TensorRT-compatible format.


    Parameters
    ----------
    input_dir : str or `os.path`
        Directory of the keras model files.
    model_name : str
        Name of the model to load.
    output_dir : str or `os.path`
        Directory to save the exported model.

    """
    # set learning phase to 'test'
    K.set_learning_phase(0)

    model = MovierecModel.load_from_dir(input_dir, model_name).model

    with K.get_session() as session:
        graph = session.graph
        with graph.as_default():
            graph_def = graph.as_graph_def()

            # freeze graph and save to file
            frozen_graph = tf_graph_util.remove_training_nodes(graph_def)
            frozen_graph = tf_graph_util.convert_variables_to_constants(
                sess=session,
                input_graph_def=frozen_graph,
                output_node_names=[out.op.name for out in model.outputs])
            # dir structure and name as expected by TensorRT
            graph_io.write_graph(
                graph_or_graph_def=frozen_graph,
                logdir=os.path.join(output_dir, '1'),
                name='model.graphdef',
                as_text=False)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Export keras model to TensortRT-compatible format.')
    parser.add_argument('-m', '--model-name', type=str, required=True,
                        help='Model name')
    parser.add_argument('-i', '--input-model-dir', type=str, required=True,
                        help='Input model directory (absolute or relative path)')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory (absolute or relative path)')
    args = parser.parse_args()

    export_keras_model_to_trt(args.input_model_dir, args.model_name, args.output_dir)
