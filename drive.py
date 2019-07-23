import tensorflow as tf
import cv2
import numpy as np
import base64

from simulator.simulator_auto_drive import SimulatorAutoDrive
import config
import models
import utils

checkpoint = tf.train.get_checkpoint_state(config.MODEL_PATH)
if checkpoint and checkpoint.model_checkpoint_path:
    inputs, keep_prob, result = models.build_model(False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Checkpoint Loaded", checkpoint.model_checkpoint_path)


        def prediction(sid, data):
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR)
            image = utils.process_image(decoded, config.INPUT_IMAGE_CROP)
            feed_dict = {inputs: [image], keep_prob: 1.0}
            pred = sess.run(result, feed_dict=feed_dict)[0]
            return pred, .1


        sim = SimulatorAutoDrive(prediction)
        sim.run()
else:
    raise FileNotFoundError("model_checkpoint_path not exist")
