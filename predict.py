import tensorflow as tf
import matplotlib.pyplot as plt

import config
import models
import utils


def visualization(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


checkpoint = tf.train.get_checkpoint_state(config.MODEL_PATH)
if checkpoint and checkpoint.model_checkpoint_path:
    batch_size = 1
    inputs, keep_prob, result = models.build_model(False)
    dataset = utils.load_training_data(batch_size=batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)

        for i in range(1):
            test_image, test_label, test_path = dataset.next_batch(augmented=False)
            feed_dict = {inputs: test_image, keep_prob: .5}

            # for m in range(batch_size):
            visualization(test_image[0])

            pred = sess.run(result, feed_dict=feed_dict)

            # for m in range(batch_size):
            print("path: {}, prediction: {}, label: {}".format(test_path[0], pred[0], test_label[0]))
