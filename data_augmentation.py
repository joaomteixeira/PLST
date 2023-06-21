import os
import tensorflow as tf
import cv2
import random

DATASET_DIR = "dataset/train"
DATASET_DEST = "augmented_dataset/train"

if not os.path.isdir(DATASET_DEST):
    os.mkdir(DATASET_DEST)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.Rescaling(1.0 / 255),
    ]
)


GOAL = 3000

# iterate over classes
for dir in os.listdir(DATASET_DIR):
    if not os.path.isdir(f"{DATASET_DEST}/{dir}"):
        os.mkdir(f"{DATASET_DEST}/{dir}")

    images = os.listdir(f"{DATASET_DIR}/{dir}")

    # write original images in new directory
    for image in images:
        img = cv2.imread(f"{DATASET_DIR}/{dir}/{image}")
        cv2.imwrite(f"{DATASET_DEST}/{dir}/{image}", img)

    number_original = len(images)
    images_todo = GOAL - number_original

    while images_todo > 0:
        i = random.randint(0, number_original - 1)

        img = cv2.imread(f"{DATASET_DIR}/{dir}/{images[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        result = data_augmentation(img)
        result = cv2.cvtColor(result.numpy() * 255, cv2.COLOR_RGB2BGR)  # RGB to BGR
        cv2.imwrite(f"{DATASET_DEST}/{dir}/augmented{images_todo}_{images[i]}", result)

        images_todo -= 1
