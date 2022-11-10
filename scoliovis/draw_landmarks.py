# UTILITY FUNCTIONS
# Point Data Structure
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point<{self.x}, {self.y}>"

# denormalize function util
def split_XY(landmarks):
    """
    For separating landmarks.

    This function assumes that the order of landmarks is follows this order:\n
    [v1x1, v1x2, ..., v17x4, v1y1, v1y2, ..., v17y4 ] (n=136)

    Args:
        landmarks: ndarray of the landmarks.

    Returns:
        tuple(ndarray, ndarray): 
            ndarray: [v1x1, v1x2, v1x3, ..., v17x4] (68)
            ndarray: [v1y1, v1y2, v1y3, ..., v17y4] (68)

    Raises:
        None
    """
    first_half: np.ndarray = landmarks[:68]
    second_half: np.ndarray = landmarks[68:]

    return (first_half, second_half)


def denormalize(normalized_landmarks, image_X_size: int, image_Y_size: int):
    """
    For transforming normalized landmarks into pixel values.

    This function assumes that the order of normalized_landmarks is follows this order:\n
    [v1x1, v1x2, ..., v17x4, v1y1, v1y2, ..., v17y4 ] (n=136)

    Args:
        normalized_landmarks: ndarray of the normalized landmarks .
        image_X_size: Image shape in the X axis.
        image_Y_size: Image shape in the Y axis.

    Returns:
        ndarray: [v1x1, v1x2, ..., v17x4, v1y1, v1y2, ..., v17y4 ] (n=136)

    Raises:
        None
    """
    first_half, second_half = split_XY(normalized_landmarks)

    first_half = first_half * image_X_size
    second_half = second_half * image_Y_size

    denormalized_numbers = np.concatenate((first_half, second_half))
    denormalized_numbers = np.round(denormalized_numbers)
    return list(denormalized_numbers)

# to points


def to_points(landmarks):
    x_list, y_list = split_XY(landmarks)
    points: list[Point] = [Point(x, y) for x, y in zip(x_list, y_list)]
    return points


import numpy as np
import tensorflow as tf
import cv2 as cv


def draw_landmarks(image, height, width, landmarks):
    # TF Image -> PIL Image -> Mat Image (CV)
    img = np.asarray(tf.keras.utils.array_to_img(image))
    img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    # Denormalize Landmarks
    lm = denormalize(landmarks, width, height)
    lm = [int(i) for i in lm]
    points = to_points(lm)

    # Draw Landmarks
    for point in points:
        cv.drawMarker(img, (point.x, point.y), color=(0, 0, 255),
                      markerType=1, thickness=2, markerSize=20)
    
    return img
