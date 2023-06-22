# PSLT - Portuguese Sign Language Translator

This project consists on using Machine Learning models and a camera to translate portuguese sign Language to text.

<div id="readme-top"></div>

## Overview

The PSLT (Portuguese Sign Language Translator) project aims to bridge the communication gap between the deaf community and the general population by utilizing machine learning models and computer vision techniques. The project focuses on translating the Portuguese Sign Language (PSL) alphabet into text, enabling real-time interpretation of sign language gestures.

The approach taken consists of using a hand landmark recognition model, such as **holistic** from **MediaPipe**, to obtain the hand landmarks from input images or video streams. These hand landmarks represent the key points on the hand, capturing the intricate movements and gestures of sign language.
Feature generation techniques are then applied to extract relevant information from the hand landmarks, which is fed into a multilayer perceptron (MLP) for prediction.

By leveraging the power of machine learning, the project utilizes **MediaPipe** and **OpenCV** libraries in Python to detect and track hand landmarks in real-time through a camera. The MediaPipe **holistic** model provides accurate and reliable results, allowing for precise recognition of sign language gestures.

The project follows a systematic approach, starting with the generation of a dataset through a script that records user hand movements and saves corresponding images. The dataset is augmented using techniques such as random brightness, contrast, and rotation to enhance the model's robustness.

A separate script generates a CSV file containing the hand landmarks detected in each image of the dataset. This CSV file serves as the input for training the machine learning model. The model is implemented using a handmark model class and a classifier class. The handmark model extracts the hand landmarks from an image, while the classifier generates predictions based on the extracted hand landmarks.

To showcase the functionality of the project, a translator app script is provided, which records hand signs through the camera and provides real-time predictions and confidence scores for each sign. This demonstration allows users to interact with the system and witness the translation process firsthand.

Throughout the project, utility functions are utilized to preprocess the dataset, crop and resize images, and remove images where hand landmarks cannot be detected.

The project also includes a Jupyter Notebook that demonstrates the use of transfer learning with the VGG16 model, highlighting the flexibility of incorporating different pre-trained models for sign language recognition.

The PSLT project holds the potential to significantly improve communication and accessibility for the deaf community by providing an efficient and accurate Portuguese Sign Language translation system.

![hand_landmarks.png](https://mediapipe.dev/images/mobile/hand_landmarks.png)

---

### Built With

-   Python
-   mediaPipe - for getting the handmarks
-   openCV - for camera record
-   Jupyter Notebook

## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

Make sure you have at least python **3.8** or above.

### Installation

After cloning the project using the command:

```bash
git clone https://github.com/joaomteixeira/PLST.git
```

Navigate to the project directory:

```sh
cd PLST
```

Open up a terminal with python and install all the required libraries

```sh
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Folder Structure

    .
    ├── data_augmentation.py
    ├── generate_csv.py
    ├── generate_dataset.py
    ├── generate_test.py
    ├── MLP_DIST.joblib
    ├── model.py
    ├── translator_app.py
    ├── utils.py
    ├── vgg_model.ipynb
    ├── LICENSE
    └── README.md

### data_augmentation.py

This script applies data augmentation techniques to the dataset. The techniques include:

-   Random Brightness
-   Random Contrast
-   Random Rotation

### generate_csv.py

This script generates a `.csv` file containing the detected hand landmarks for each image in the dataset. The generated `.csv` file consists of **21 \* 3** columns for the positions of each hand landmark point, as well as **2** additional columns for the image path and label.

| ID        | Label | R0x | R0y | R0z | ... | R20x | R20y | R20z |
| :-------- | :---- | :-- | :-- | :-- | :-- | :--- | :--- | :--- |
| `A01.png` | `A`   | 0.1 | 0.1 | 0.1 | ... | 0.9  | 0.9  | 0.9  |

### generate_dataset.py

This script can be used to generate the dataset. It records each movement made by the user through the camera and saves the corresponding images. To generate the dataset, you need to set the `LABEL` variable to the letter you are recording and the `BASE_DIR` variable to the directory where you want to save the images. Optionally, you can adjust the `TIME_BETWEEN_IMAGES` variable to change the image capture rate.

### generate_test.py

This script move images from the **training dataset** to the **test dataset**.

### MLP_DIST.joblib

Model that perform the best results in the test dataset (**88%**).

### model.py

This module contains two **Classes**:

-   HandmarkModel

-   Classifier

The **HandmarkModel** is responsible for obtaining the hand landmarks from an image using the **holistic** model from **MediaPipe**.

The **Classifier** is responsible for generating a prediction given a set of points corresponding to the hand landmark outputted by the previous model.

### translator_app.py

This script demonstrates the functionality of the project. It records the user's hand signs and provides real-time predictions and confidence scores for each sign.

### utils.py

This module contains utility functions used throughout the project:

| function                     | description                                                                                                                                  |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `find_rect`                  | Given a hand landmark, the function outputs the **top left** and **bottom right** corners of the rectangle that surrounds the hand landmarks |
| `crop_and_resize`            | Given a hand landmarks, this funciton crops an image where the hand landmarks are positioned and resizes the image to a given shape          |
| `pre_process_dataset`        | Transforms each image in a given folder to a cropped image where only the hand is visible.                                                   |
| `check_image_can_be_cropped` | Identifies which image in a given folder can be cropped based on the hand landmarks                                                          |
| `remove_images_not_detected` | Removes images where the hand landmarks cannot be detected                                                                                   |

### vgg_model.ipynb

This Jupyter Notebook demonstrates the use of **transfer learning**. It utilizes the pre-trained **VGG16** model, but it can be easily adapted to use another pre-trained model.

## Authors

-   [@Joao-Amoroso](https://www.github.com/Joao-Amoroso)
-   [@joaomteixeira](https://www.github.com/joaomteixeira)

## License

[MIT](https://choosealicense.com/licenses/mit/)
