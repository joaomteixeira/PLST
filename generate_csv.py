import cv2
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from model import HandmarkModel

# creates model
handmarks_model = HandmarkModel(static=True)


images = []
PATH = "../dataset/train"
# PATH = "../test"
OUTPUT_FILENAME = "handmarks.csv"


folders = os.listdir(PATH)

# loop through images in dataset folder and fed them to the model to capture the landmarks
for i in tqdm(range(len(folders)), desc=" outer", position=0):
    letter_path = folders[i]
    imags = os.listdir(f"{PATH}/{letter_path}")

    for j in tqdm(range(len(imags)), desc=" inner loop", position=1, leave=False):
        image_path = imags[j]
        img = cv2.imread(f"{PATH}/{letter_path}/{image_path}")

        # Converting the from BGR to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = handmarks_model.process(image)
        image.flags.writeable = True
        # handmarks_model.draw_handmarks(image, results)
        images.append([image_path, letter_path, results])

columns = ["ID", "Label"]


# add Right hand labels
for i in range(21):
    columns.append(f"R{i}x")
    columns.append(f"R{i}y")
    columns.append(f"R{i}z")


count = 0
for image in images:
    if image[2].right_hand_landmarks != None or image[2].left_hand_landmarks != None:
        count += 1
print(f"{count}/{len(images)}")


data = []
for image in images:
    image_path = image[0]
    label = image[1]
    results = image[2]
    right_handmarks = []

    if results.right_hand_landmarks != None:
        for point in results.right_hand_landmarks.landmark:
            right_handmarks.append(point.x)
            right_handmarks.append(point.y)
            right_handmarks.append(point.z)
    else:
        continue

    data.append([image_path] + [label] + right_handmarks)


df = pd.DataFrame(data, columns=columns)

df.to_csv(OUTPUT_FILENAME, index=False)
