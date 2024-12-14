import numpy as np
import quaternion


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
