import numpy as np
import matplotlib.pyplot as plt

import os
import config
import csv
import random
import cv2

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3


def static_vars(**args):
    def decorate(fn):
        for var in args:
            setattr(fn, var, args[var])
        return fn

    return decorate


def process_image(images, crop):
    # 裁剪天空
    crop = images[crop[0]:crop[1], :, :]
    # 调整尺寸至模型输入尺寸
    resize = cv2.resize(crop, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    # 转换色彩空间到YUV
    yuv = cv2.cvtColor(resize, cv2.COLOR_BGR2YUV)
    # 归一化
    return yuv / 127.5 - 1.0


def augment_image(images, steering):
    images, steering = random_flip(images, steering)
    images, steering = random_translate(images, steering, 50, 10)
    images = random_shadow(images)
    images = random_brightness(images)

    return images, steering


def random_flip(image, steering):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def load_training_data():
    image_paths = []
    labels = []

    with open(os.path.join(config.DATASET_PATH, config.DATASET_META_FILE), 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for center_path, left_path, right_path, steering, _, _, _ in reader:
            steering_angle = float(steering.strip())
            image_paths.extend([center_path.strip(), left_path.strip(), right_path.strip()])
            labels.extend([steering_angle, steering_angle + config.ANGLE_DELTA_CORRECTION_LEFT,
                           steering_angle + config.ANGLE_DELTA_CORRECTION_RIGHT])
    return image_paths, labels


@static_vars(offset_train=0, order_train=[])
def next_batch(images, labels):
    image_train_count = len(images)
    if image_train_count != len(labels):
        raise ValueError("Unmatched count of image_train and label_train")
    if image_train_count < config.BATCH_SIZE:
        raise ValueError("Count of image_train is less than batch size")
    if image_train_count != len(next_batch.order_train):
        next_batch.order_train = list(range(image_train_count))
    if next_batch.offset_train + config.BATCH_SIZE > image_train_count:
        next_batch.offset_train = 0
        random.shuffle(next_batch.order_train)

    image_train_batch = []
    label_train_batch = []
    for _ in range(config.BATCH_SIZE):
        file_path = os.path.join(config.DATASET_PATH, images[next_batch.offset_train])
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        steering = labels[next_batch.offset_train]
        if np.random.rand() < 0.5:
            image, steering = augment_image(image, steering)
        image_train_batch.append(process_image(image, config.INPUT_IMAGE_CROP))
        label_train_batch.append(steering)
        next_batch.offset_train += 1

    return image_train_batch, label_train_batch


@static_vars(offset_train=0, order_train=[])
def next_test_batch(images, labels, batch_size):
    image_train_count = len(images)
    if image_train_count != len(labels):
        raise ValueError("Unmatched count of image_train and label_train")
    if image_train_count < batch_size:
        raise ValueError("Count of image_train is less than batch size")
    if image_train_count != len(next_test_batch.order_train):
        next_test_batch.order_train = list(range(image_train_count))
    if next_test_batch.offset_train + batch_size > image_train_count:
        next_test_batch.offset_train = 0
        random.shuffle(next_test_batch.order_train)

    image_train_batch = []
    label_train_batch = []
    for _ in range(batch_size):
        file_path = os.path.join(config.DATASET_PATH, images[next_test_batch.offset_train])
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        steering = labels[next_test_batch.offset_train]
        image_train_batch.append(process_image(image, config.INPUT_IMAGE_CROP))
        label_train_batch.append(steering)
        next_test_batch.offset_train += 1

    return image_train_batch, label_train_batch


def visualization(image):
    plt.figure()
    plt.imshow(image)
    plt.show()
