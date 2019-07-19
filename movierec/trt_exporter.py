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


CONFIG_FILE_NAME = "config.pbtxt"
MODEL_VERSION_DIR_NAME = '1'
MAX_BATCH_SIZE = '1024'
# some data types mappings from
# https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#section-datatypes
DATA_TYPES_MAP = {
    tf.int16: "TYPE_INT16",
    tf.int32: "TYPE_INT32",
    tf.int64: "TYPE_INT64",
    tf.float16: "TYPE_FP16",
    tf.float32: "TYPE_FP32",
    tf.float64: "TYPE_FP64",
}


def export_keras_model_to_trt(input_dir, model_name, output_dir):
    """
    Export a saved keras model to a TensorRT-compatible model. Steps: load keras model from file,
    freeze and optimize graph for inference, save in TensorRT-compatible format.

    See:
    https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_repository.html#tensorflow-models

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
            output_file_path = os.path.join(output_dir, model_name, MODEL_VERSION_DIR_NAME)
            graph_io.write_graph(
                graph_or_graph_def=frozen_graph,
                logdir=output_file_path,
                name='model.graphdef',
                as_text=False)
    logging.info("Saved graph def file to {}".format(output_file_path))
    # write config, setting only one output
    _write_config_file(output_dir, model_name, model.inputs, [model.outputs[0]])


def _write_config_file(output_dir, model_name, inputs, outputs):
    """
    Write model configuration file.

    See
    https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html

    Parameters
    ----------
    output_dir : str or `os.path`
        Directory of the the exported model.
    model_name : str
        Name of the model
    inputs : List of `tf.Tensor`
        Model inputs.
    outputs: List of `tf.Tensor`
        Model outputs.

    """
    config_str = 'name: "{}"\n' \
                 'platform: "tensorflow_graphdef"\n' \
                 'max_batch_size: {}\n' \
                 'input [\n'.format(model_name, MAX_BATCH_SIZE)

    def get_tensors_str_list(tensors):
        tensors_str_list = []
        for tensor in tensors:
            name = tensor.op.name
            data_type = DATA_TYPES_MAP[tensor.dtype]
            dims = ' '.join([str(dim.value) for dim in tensor.shape[1:]])

            tensors_str_list.append('  {{\n'
                                    '    name: "{}"\n'
                                    '    data_type: {}\n'
                                    '    dims: [ {} ]\n'
                                    '\n  }}'.format(name, data_type, dims))
        return tensors_str_list

    config_str += ','.join(get_tensors_str_list(inputs))
    config_str += '\n]\n' \
                  'output [\n'
    config_str += ','.join(get_tensors_str_list(outputs))
    config_str += '\n]\n'

    # write to file
    config_file_path = os.path.join(output_dir, model_name, CONFIG_FILE_NAME)
    with open(config_file_path, 'w') as f_out:
        f_out.write(config_str)
    logging.info("Saved config file to {}".format(config_file_path))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Export keras model to TensortRT-compatible format.')
    parser.add_argument('-m', '--model-name', type=str, required=True,
                        help='Model name')
    parser.add_argument('-i', '--input-model-dir', type=str, required=True,
                        help='Input model directory (absolute or relative path)')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory (absolute or relative path)')
    parser.add_argument('-l', '--log-level', type=str, default='INFO',
                        help='Log level (default: INFO).')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.log_level))

    export_keras_model_to_trt(args.input_model_dir, args.model_name, args.output_dir)
