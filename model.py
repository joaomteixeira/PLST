import mediapipe as mp

from joblib import load
import numpy as np


class HandmarkModel:
    def __init__(self, static=False) -> None:
        # Grabbing the Holistic Model from Mediapipe and
        # Initializing the Model
        self.mp_holistic = mp.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=static,
        )

        # Initializing the drawing utils for drawing the facial landmarks on image
        self.mp_drawing = mp.solutions.drawing_utils

    def process(self, image):
        """
        computes handmarks

        Parameters:
            image: an numpy array
        """
        return self.holistic_model.process(image)

    def draw_handmarks(self, image, results):
        """
        draws handmarks in-place

        Parameters:
            image: an numpy array that will be overwritten
            results: handmarks
        """

        # Drawing Right hand Land Marks
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        # Drawing Left hand Land Marks
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )


class Classifier:
    def __init__(self, feature_generation_type=0):
        """
        feature_generation_type: 0 dist, 1 dist + angle
        """
        self.clf = load("MLP_DIST.joblib")
        self.gen_type = feature_generation_type

    def predict(self, results):
        """
        results - outputs from Handmark Model
        returns: class predicted and a float between 0 and 1 representing the confidence
        """
        handmarks = results.right_hand_landmarks

        if handmarks is None:
            return None, None

        # flat array and generate features

        points = []

        for point in handmarks.landmark:
            points.append(point.x)
            points.append(point.y)
            points.append(point.z)

        points = np.array(points)

        if self.gen_type == 0:
            points = self.distance(points)

        points = points.reshape(1, len(points))

        pred = self.clf.predict(points)
        prob = self.clf.predict_proba(points)
        m = np.amax(prob)

        return pred[0], m

    def distance(self, points):
        """
        points: set of 21 points with 3D coordinates
        return the points concatenated with the distances between each finger tip
        """
        finger_points = (0, 4, 8, 12, 16, 20)

        dist = []

        for i in range(len(finger_points)):
            point = finger_points[i]
            i_position = points[point * 3 : point * 3 + 3]

            for j in range(i + 1, len(finger_points)):
                point2 = finger_points[j]
                j_position = points[point2 * 3 : point2 * 3 + 3]

                c = np.linalg.norm(j_position - i_position, axis=0, ord=2)
                dist.append(c)
        dist = np.array(dist)
        result = np.concatenate([points, dist])

        return result
