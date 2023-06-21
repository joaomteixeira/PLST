import os
import shutil
import random



keys = os.listdir(f"pre_process_dataset/train")


path_dest = "test_dataset_generated"

if not os.path.isdir(path_dest):
    os.mkdir(path_dest)

for key in keys:
    images = os.listdir(f"pre_process_dataset/train/{key}")
    if not os.path.isdir(f"{path_dest}/{key}"):
        os.mkdir(f"{path_dest}/{key}")
    for _ in range(5):
        i = random.randint(0, len(images) - 1)
        shutil.copy2(f"pre_process_dataset/train/{key}/{images[i]}", f"{path_dest}/{key}/{images[i]}")
