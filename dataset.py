import random
import os
import cv2
import numpy as np

import utils
import config


class Dataset:
    # 分页数量
    batch_size = 100
    # 偏移
    offset = 0
    # 顺序
    orders = []

    # 图片路径列表
    images_path = []
    # 标签列表
    labels = []

    def __init__(self, images_path, labels, batch_size=100):

        image_count = len(images_path)
        if image_count != len(labels):
            raise ValueError("Unmatched count of images and labels")
        if image_count < batch_size:
            raise ValueError("Count of images is less than batch size")

        self.images_path = images_path
        self.labels = labels
        self.batch_size = batch_size
        self.orders = list(range(image_count))
        random.shuffle(self.orders)

    def next_batch(self, augmented=True):
        if self.offset + self.batch_size > len(self.images_path):
            self.offset = 0
            random.shuffle(self.orders)

        image_batch = []
        label_batch = []
        path_batch = []
        for _ in range(self.batch_size):
            file_path = os.path.join(config.DATASET_PATH, self.images_path[self.orders[self.offset]])
            path_batch.append(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            steering = self.labels[self.orders[self.offset]]
            if augmented and np.random.rand() < 0.5:
                image, steering = utils.augment_image(image, steering)
            image_batch.append(utils.process_image(image, config.INPUT_IMAGE_CROP))
            label_batch.append(steering)
            self.offset += 1

        return image_batch, label_batch, path_batch
