import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

import config

checkpoint = tf.train.get_checkpoint_state(config.MODEL_PATH)

if checkpoint and checkpoint.model_checkpoint_path:
    output_graph = os.path.join(config.MODEL_PATH, "frozen_model.pb")
    output_node_name = "predict/result"

    saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + ".meta", clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_ref = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_ref, output_node_name.split(','))

        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
