import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

import config
import utils

plt.figure()

img = cv2.imread(os.path.join(config.DATASET_PATH, "driving_images/center_2019_07_17_19_51_34_642.jpg"),
                 cv2.IMREAD_COLOR)

plt.subplot(311)
plt.imshow(img)

img, _ = utils.augment_image(img, 0.5)
plt.subplot(312)
plt.imshow(img)

img = utils.process_image(img, config.INPUT_IMAGE_CROP)
plt.subplot(313)
plt.imshow(img)
plt.show()
