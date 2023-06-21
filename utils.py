import cv2

import json
import os
from tqdm import tqdm
from model import HandmarkModel

import warnings


def find_rect(handlandmark, width, height):
    """
    handlandmarks: Handmarkmodel output
    width: width
    height: height
    return -> (top left point, bottom right point)
    """
    minx = 2
    maxx = -1
    miny = 2
    maxy = -1
    for point in handlandmark.landmark:
        minx = min(point.x, minx)
        miny = min(point.y, miny)
        maxx = max(point.x, maxx)
        maxy = max(point.y, maxy)

    start_point = (int(minx * width), int(miny * height))
    end_point = (int(maxx * width), int(maxy * height))

    return start_point, end_point


def crop_and_resize(image, handlandarmks, padding=60, resize=(224, 224)):
    """
    image: HEIGHTxWIDTHx3
    handlandmarks: Handmarkmodel output
    padding: padding of crop
    returns -> List[images]
    """

    height, width, _ = image.shape
    res = []

    # if right hand detected
    if handlandarmks.right_hand_landmarks:
        # find top-left point and bottom-right point that defines the rectangle
        start, end = find_rect(handlandarmks.right_hand_landmarks, width, height)

        # add padding
        start = (max(start[0] - padding, 0), max(start[1] - padding, 0))

        end = (
            min(end[0] + padding, width),
            min(end[1] + padding, height),
        )

        # crop image
        cropped_image = image[start[1] : end[1], start[0] : end[0], :]

        # resize image
        resized_image = cv2.resize(cropped_image, resize)

        res.append(resized_image)

    # if left hand detected
    if handlandarmks.left_hand_landmarks:
        # find top-left point and bottom-right point that defines the rectangle

        start, end = find_rect(handlandarmks.left_hand_landmarks, width, height)

        # add padding
        start = (max(start[0] - padding, 0), max(start[1] - padding, 0))
        end = (
            min(end[0] + padding, width),
            min(end[1] + padding, height),
        )

        # crop image
        cropped_image = image[start[1] : end[1], start[0] : end[0], :]

        # resize image
        resized_image = cv2.resize(cropped_image, resize)

        res.append(resized_image)

    return res


def pre_process_dataset(path, output_dir="./pre_process_dataset", train=True):
    """
    Generates cropped images with correct dataset
    path: path to folder above letters
    """

    # checks if path exists and if is a directory
    if not os.path.isdir(path):
        warnings.warn("{path} must be a directory")
        raise Exception("{path} must be a directory")

    path = path + "/" + ("train" if train else "test")

    # checks if path exists and if is a directory
    if not os.path.isdir(path):
        warnings.warn(f"{path} is not a directory")
        raise Exception(f"{path} is not a directory")

    # checks if outputdir exists and if not, it is created
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # checks if outputdir is a directory
    if not os.path.isdir(output_dir):
        warnings.warn("Output is not a folder")
        raise Exception("Output is not a folder")

    output_dir = output_dir + "/" + ("train" if train else "test")

    # checks if new outputdir exists and if not, it is created
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # checks if new outputdir exists and if not, it is created
    if not os.path.isdir(output_dir):
        warnings.warn("Output is not a folder")
        raise Exception("Output is not a folder")

    # static is true to not treat images as a video stream
    handmarks_model = HandmarkModel(static=True)

    childs_folder = os.listdir(path)

    # iterates through label folders
    for i in tqdm(range(len(childs_folder)), desc=" outer", position=0):
        path_to_child = f"{path}/{childs_folder[i]}"

        if not os.path.isdir(path_to_child):
            warnings.warn(f"Everything inside {path} must be a folder")
            raise Exception(f"Everything inside {path} must be a folder")

        images = os.listdir(path_to_child)

        if not os.path.exists(output_dir + "/" + childs_folder[i]):
            os.mkdir(output_dir + "/" + childs_folder[i])

        if not os.path.isdir(output_dir + "/" + childs_folder[i]):
            warnings.warn(f"{output_dir}/{childs_folder[i]} is not a folder")
            raise Exception(f"{output_dir}/{childs_folder[i]} is not a folder")

        # iterates through images
        for j in tqdm(range(len(images)), desc=" inner loop", position=1, leave=False):
            path_to_image = f"{path}/{childs_folder[i]}/{images[j]}"

            if os.path.isdir(path_to_image):
                warnings.warn(
                    f"Everything inside {path}/{childs_folder[i]} must be an image"
                )
                raise Exception(
                    f"Everything inside {path}/{childs_folder[i]} must be an image"
                )

            if images[j].split(".")[-1] not in ["png", "jpg", "jpeg"]:
                warnings.warn(
                    f"Everything inside {path}/{childs_folder[i]} must be an image "
                )
                raise Exception(
                    f"Everything inside {path}/{childs_folder[i]} must be an image "
                )

            # get image
            img = cv2.imread(path_to_image)

            # Converting the from BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            rgb_img.flags.writeable = False
            results = handmarks_model.process(rgb_img)

            rgb_img.flags.writeable = True

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            images_croped = crop_and_resize(rgb_img, results)

            splited = images[j].split(".")

            for k, image_croped in enumerate(images_croped):
                output_name = splited[:-1] + [str(k)] + splited[-1:]
                output_name = "_".join(output_name[:-1]) + "." + output_name[-1]
                cv2.imwrite(
                    f"{output_dir}/{childs_folder[i]}/{output_name}", image_croped
                )


def check_image_can_be_cropped(path="./dataset", train=True):
    """
    This function goes through each letter in the give path, and check
    each images can be cropped. It writes to a json file all the images
    that could be cropped in each letter folder.

    path: path to folder above letters
    """
    path = path + "/" + ("train" if train else "test")

    folders = os.listdir(path)

    # static is true to not treat images as a video stream
    handmarks_model = HandmarkModel(static=True)

    res = {}

    # iterates through label folders
    for i in tqdm(range(len(folders)), desc=" outer", position=0):
        path_to_child = f"{path}/{folders[i]}"

        images = os.listdir(path_to_child)
        count = 0
        images_not_recognized = []
        # iterates through images
        for j in tqdm(range(len(images)), desc=" inner loop", position=1, leave=False):
            path_to_image = f"{path_to_child}/{images[j]}"

            # get image
            img = cv2.imread(path_to_image)

            # Converting the from BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            rgb_img.flags.writeable = False
            results = handmarks_model.process(rgb_img)

            rgb_img.flags.writeable = True

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            if (
                results.left_hand_landmarks is not None
                or results.right_hand_landmarks is not None
            ):
                images_not_recognized.append(images[j])
                count += 1

        res[folders[i]] = (count, len(images), images_not_recognized)

    for key in res:
        print(f"{key} -> {res[key][0]}/{res[key][1]}")

    with open("results.json", "w") as f:
        json.dump(res, f)


def remove_images_not_detected(path="./dataset/train", json_filename="results.json"):
    """
    This function removes every image from the letter present in the path that
    the Hand landmark recognition model couldn't recognize.

    path: path to the folder that contains letters
    json_filename: The name of the file that contains the images that the hand landmark model couldn't recognize
    """

    data = None
    with open(json_filename, "r") as f:
        data = json.load(f)
    for key in data.keys():
        print(key)
        if data[key][1] - data[key][0] == 0:
            continue
        images = os.listdir(f"{path}/{key}")
        count = 0
        for img in images:
            if img not in data[key][2]:
                os.remove(f"{path}/{key}/{img}")
                count += 1
        print(f"Removed -> {count}")
