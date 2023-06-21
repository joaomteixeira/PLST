import os
import sys
import time

import cv2
import matplotlib.pyplot as plt

from model import HandmarkModel

USE_MODEL = True

LABEL = "x"
BASE_DIR = "../dataset"

# number of images wanted for letter
IMAGES_WANTED = 750

# seconds
TIME_BETWEEN_IMAGES = 0.25
TIME_BETWEEN_IMAGES = 0.15


if USE_MODEL:
    TIME_BETWEEN_IMAGES = 0.15


def generate_directories(PATH):
    """
    PATH -> receive the path for the letter in training dataset
    returns number of images present in that directory
    """
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
        n_images = 0

    else:
        n_images = len(
            [
                entry
                for entry in os.listdir(PATH)
                if os.path.isfile(os.path.join(PATH, entry))
            ]
        )
    return n_images


def take_photos(n_images):
    """
    n_images -> number of images for that letter already done
    returns new images to be appended
    """
    if USE_MODEL:
        handmarks_model = HandmarkModel(static=True)
    images = []

    # define default camera
    vid = cv2.VideoCapture(0)

    # Initializing variables needed to calculate FPS
    previousTime = 0
    currentTime = 0

    # small delay to get time to do the sign
    time.sleep(3.5)

    # keeps track of number of photos taken
    count = 0

    # number of images wanted to take for session
    IMAGES_TO_TAKE = IMAGES_WANTED - n_images

    while vid.isOpened() and count < IMAGES_TO_TAKE:
        count += 1
        time.sleep(TIME_BETWEEN_IMAGES)
        # Capture the video frame by frame
        ret, frame = vid.read()
        if frame is None:
            print("Breaking")
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Converting the from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if USE_MODEL:
            image.flags.writeable = False
            results = handmarks_model.process(image)
            image.flags.writeable = True

            flag = (
                results.right_hand_landmarks is None
                and results.left_hand_landmarks is None
            )

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Converting back the RGB image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not USE_MODEL or not flag:
            images.append(image.copy())

        # print(int(count / IMAGES_TO_TAKE))
        # Displaying FPS on the image
        text = (
            str(int(fps))
            + " FPS took "
            + str(count)
            + " "
            + str(int(count / IMAGES_TO_TAKE * 100))
            + "%"
        )

        if USE_MODEL:
            text = text + str("Apanhou" if not flag else "Não")
        cv2.putText(
            image,
            text,
            (10, 70),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display the resulting frame
        cv2.imshow("frame", image)

        # use 'q' key to close the program
        if cv2.waitKey(1) == ord("q") or cv2.waitKey(1) == ord("Q"):
            break

    # After the loop release the vid object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    return images


if __name__ == "__main__":
    PATH = f"{BASE_DIR}/train/{LABEL.upper()}"
    n_images = generate_directories(PATH)

    images = take_photos(n_images)

    for i in range(len(images)):
        cv2.imwrite(f"{PATH}/{LABEL}_{n_images + i + 1}.jpg", images[i])
