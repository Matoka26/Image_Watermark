import numpy as np
import quaternion
import cv2

def rgb_to_quat(img: np.ndarray) -> quaternion:
    real_layer = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    new_img = np.concatenate((real_layer, img), axis=2)
    return quaternion.as_quat_array(new_img)


def quat_to_rgb(quat_img: quaternion) -> np.ndarray:
    quat_array = quaternion.as_float_array(quat_img)

    red = quat_array[:, :, 1]
    green = quat_array[:, :, 2]
    blue = quat_array[:, :, 3]

    rgb_img = np.stack((red, green, blue), axis=2).astype(np.uint8)

    return rgb_img


def rgb_to_binary(img: np.ndarray) -> np.ndarray:
    img = cv2.imread('lena.png', 2)

    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # converting to its binary form
    bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)