import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import cv2
import numpy as np

import utils
import config

node_name_prefix = "nvidia"
input_node_name = node_name_prefix + "/inputs:0"
keep_prob_node_name = node_name_prefix + "/keep_prob:0"
prediction_node_name = node_name_prefix + "/predict/result:0"
output_graph = os.path.join(config.MODEL_PATH, config.FREEZE_MODEL_NAME + ".pb")


def visualization(image, processed_image):
    plt.figure()
    plt.subplot(211)
    plt.imshow(image)
    plt.subplot(212)
    plt.imshow(processed_image)
    plt.show()


with tf.gfile.GFile(output_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    # fix nodes
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        name=node_name_prefix,
                        input_map=None,
                        return_elements=None,
                        op_dict=None,
                        producer_op_list=None
                        )

    for op in graph.get_operations():
        print(op.name, op.values())

    x = graph.get_tensor_by_name(input_node_name)
    keep_prob = graph.get_tensor_by_name(keep_prob_node_name)
    y = graph.get_tensor_by_name(prediction_node_name)

    dataset = utils.load_training_data(batch_size=1)
    with tf.Session(graph=graph) as sess:
        # image_count = len(image_paths)
        # rand = random.randint(1, image_count - 1)
        # image_path = image_paths[rand]
        #
        # image = cv2.imread(os.path.join(config.DATASET_PATH, image_path), cv2.IMREAD_COLOR)
        # processed_image = utils.process_image(image, config.INPUT_IMAGE_CROP)

        image_batch, label_batch = dataset.next_batch(augmented=False)

        visualization(image_batch[0], image_batch[0])

        # reshaped_input = np.reshape(image_batch[0], (1, 66, 200, 3))

        sess.run(tf.global_variables_initializer())
        print(sess.run(y, feed_dict={x: image_batch, keep_prob: .5}))
        # print(sess.run(y, feed_dict={x: reshaped_input, keep_prob: .5}))
