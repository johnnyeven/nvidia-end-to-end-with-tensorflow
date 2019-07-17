import os
import config
import csv
import random
import cv2


def static_vars(**args):
    def decorate(fn):
        for var in args:
            setattr(fn, var, args[var])
        return fn

    return decorate


def pre_process_image(images, crop):
    yuv = cv2.cvtColor(cv2.resize(images[crop[0]:crop[1], crop[2]:crop[3], :], (200, 66), cv2.INTER_AREA),
                       cv2.COLOR_BGR2YUV)
    return yuv / 127.5 - 1.0


def load_training_data():
    image_paths = []
    labels = []

    with open(os.path.join(config.DATASET_PATH, config.DATASET_META_FILE), 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for path, steering in reader:
            steering_angle = float(steering.strip())
            image_paths.append(path)
            labels.append(steering_angle)
    return image_paths, labels


@static_vars(offset=0, orders=[])
def next_batch(image_paths, labels):
    image_count = len(image_paths)
    if image_count != len(labels):
        raise ValueError("Unmatched count of images and labels")
    if image_count < config.BATCH_SIZE:
        raise ValueError("Count of images is less than batch size")
    if image_count != len(next_batch.orders):
        next_batch.orders = list(range(image_count))
    if next_batch.offset + config.BATCH_SIZE > image_count:
        next_batch.offset = 0
        random.shuffle(next_batch.orders)

    image_batch = []
    label_batch = []
    for _ in range(config.BATCH_SIZE):
        file_path = os.path.join(config.DATASET_PATH, image_paths[next_batch.offset])
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image_batch.append(pre_process_image(image, config.INPUT_IMAGE_CROP))
        label_batch.append(labels[next_batch.offset])
        next_batch.offset += 1

    return image_batch, label_batch
