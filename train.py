import tensorflow as tf
from sklearn.model_selection import train_test_split
import signal

from utils import *
import models

inputs, keep_prob, result = models.build_model()
labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")

loss = models.build_loss(labels=labels, result=result, regularizer_weight=config.REGULARIZER_WEIGHT,
                         regularized_vars=tf.trainable_variables())

training_step = tf.Variable(0, trainable=False, name="training_step")
train = tf.train.AdamOptimizer(1e-5).minimize(loss, global_step=training_step)

image_paths, image_labels = load_training_data()

tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

exit_signal = False


def exit_handler(*args):
    exit_signal = True


signal.signal(signal.SIGINT, exit_handler)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step = 0
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(config.MODEL_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint detected", checkpoint.model_checkpoint_path)
    log_writer = tf.summary.FileWriter(config.LOG_PATH, sess.graph)

    for _ in range(config.MAX_STEPS):
        image_train, label_train = next_batch(image_paths, image_labels)

        _, step, result_batch, loss_batch, summary = sess.run([train, training_step, result, loss, summary_op],
                                                              feed_dict={inputs: image_train, labels: label_train,
                                                                         keep_prob: config.KEEP_PROB})

        if step % config.LOG_INTERVAL == 0:
            print(
                "Step: {}, Loss: {}, Prediction: {}, Label: {}".format(step, loss_batch, result_batch[0],
                                                                       label_train[0]))
            log_writer.add_summary(summary, global_step=step)
        if exit_signal or step % config.SAVE_INTERVAL == 0:
            saver.save(sess, os.path.join(config.MODEL_PATH, config.MODEL_NAME), global_step=step)
            print("Model saved")
            if exit_signal:
                break

    saver.save(sess, os.path.join(config.MODEL_PATH, config.MODEL_NAME), global_step=step)
