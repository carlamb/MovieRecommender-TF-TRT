""" Export models. """

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import os
from tensorflow.python.framework.graph_io import write_graph


def export_keras_model_to_trt(model_file, output_dir):
    """
    Export a saved keras model to a TensorRT-compatible model. Steps: load keras model from file,
    freeze and optimize graph for inference, save in TensorRT-compatible format.


    Parameters
    ----------
    model_file : str or `os.path`
        File name of the keras model to export.
    output_dir : str or `os.path`
        Directory to save the exported model.

    """

    model = load_model(model_file)
    session = K.get_session()

    # set learning phase to 'test'
    K.set_learning_phase(0)

    print(model.outputs)
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=session,
            input_graph_def=session.graph.as_graph_def(),
            output_node_names=[out.op.name for out in model.outputs])
    tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)

    # write to dir following TensortRT naming requirements
    write_graph(
            graph_or_graph_def=frozen_graph,
            logdir=os.path.join(output_dir, '1'),
            name='model.graphdef',
            as_text=False)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Export keras model to TensortRT-compatible format.')
    parser.add_argument('-i', '--input-model-file', type=str, required=True,
                        help='Input model file name (absolute or relative path)')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory (absolute or relative path)')
    args = parser.parse_args()

    export_keras_model_to_trt(args.input_model_file, args.output_dir)
