import tensorflow as tf


def conv_layer(name, inputs, in_channels, out_channels, ksize, stride, trainable=True, padding="SAME",
               with_elu=True,
               with_batch_normal=True, decay=0.9, epsilon=1e-5):
    if not hasattr(ksize, '__len__'):
        ksize = [ksize, ksize]
    if not hasattr(stride, '__len__'):
        stride = [stride, stride]
    with tf.variable_scope(name):
        kernel = tf.get_variable('weight', [ksize[0], ksize[1], in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=0.1),
                                 trainable=trainable)
        biases = tf.get_variable('biases', dtype=tf.float32, shape=[out_channels], trainable=trainable,
                                 initializer=tf.constant_initializer(0, dtype=tf.float32))
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride[0], stride[1], 1], padding=padding)
        result = tf.nn.bias_add(conv, biases)

        if with_batch_normal:
            result = tf.contrib.layers.batch_norm(result, decay=decay, epsilon=epsilon, updates_collections=None,
                                                  center=True, scale=True, trainable=True, is_training=trainable)
        if with_elu:
            result = tf.nn.elu(result)
    return result, kernel, biases


def max_pool_layer(name, inputs, ksize, strides, padding="SAME"):
    with tf.variable_scope(name):
        result = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding)
    return result


def fc_layer(name, inputs, in_channels, out_channels, trainable=True, with_elu=True, with_batch_normal=True, decay=0.9,
             epsilon=1e-5):
    with tf.variable_scope(name):
        weights = tf.get_variable('weight', [in_channels, out_channels], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                  trainable=trainable)
        biases = tf.get_variable('biases', [out_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                 trainable=trainable)
        result = tf.matmul(inputs, weights) + biases
        if with_batch_normal:
            result = tf.contrib.layers.batch_norm(result, decay=decay, epsilon=epsilon, updates_collections=None,
                                                  center=True, scale=True, trainable=True, is_training=trainable)
        if with_elu:
            result = tf.nn.elu(result)
    return result, weights, biases


def build_model(trainable=True):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    inputs = tf.placeholder(tf.float32, [None, 66, 200, 3], 'inputs')

    conv1, w1, b1 = conv_layer('conv1', inputs, 3, 24, [5, 5], [2, 2], padding='VALID', trainable=trainable)
    conv2, w2, b2 = conv_layer('conv2', conv1, 24, 36, [5, 5], [2, 2], padding='VALID', trainable=trainable)
    conv3, w3, b3 = conv_layer('conv3', conv2, 36, 48, [5, 5], [2, 2], padding='VALID', trainable=trainable)
    conv4, w4, b4 = conv_layer('conv4', conv3, 48, 64, [3, 3], [1, 1], padding='VALID', trainable=trainable)
    conv5, w5, b5 = conv_layer('conv5', conv4, 64, 64, [3, 3], [1, 1], padding='VALID', trainable=trainable)

    drop = tf.nn.dropout(conv5, rate=1 - keep_prob)

    fc6, w6, b6 = conv_layer('fc1', drop, 64, 1164, [1, 18], [1, 1], padding='VALID', trainable=trainable)
    fc7, w7, b7 = conv_layer('fc2', fc6, 1164, 100, [1, 1], [1, 1], padding='VALID', trainable=trainable)
    fc8, w8, b8 = conv_layer('fc3', fc7, 100, 50, [1, 1], [1, 1], padding='VALID', trainable=trainable)
    fc9, w9, b9 = conv_layer('fc4', fc8, 50, 10, [1, 1], [1, 1], padding='VALID', trainable=trainable)
    fc10, w10, b10 = conv_layer('fc5', fc9, 10, 1, [1, 1], [1, 1], padding='VALID', trainable=trainable,
                                with_batch_normal=False, with_elu=False)

    with tf.variable_scope('predict'):
        result = tf.reshape(fc10, [-1], name="result")

    return inputs, keep_prob, result


def build_loss(labels, result, regularizer_weight, regularized_vars):
    with tf.variable_scope("mse_loss"):
        mse = tf.reduce_mean(tf.square(labels - result))
    with tf.variable_scope("l2_loss"):
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in regularized_vars])
    with tf.variable_scope("total_loss"):
        loss = mse + l2 * regularizer_weight
    return loss
