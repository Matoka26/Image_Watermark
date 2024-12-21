import cv2
import matplotlib.pyplot as plt
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


def break_image_to_blocks(img: np.ndarray, block_size: int=4) -> np.ndarray:
    i_lim, j_lim = img.shape[:2]
    blocks = []

    try:
        try:
            is_gray = img.shape[2] == 1
        except IndexError:
            is_gray = 1
    except ValueError:
        is_gray = 0

    if is_gray:
        img = np.pad(
            img,
            [
                (0, (block_size - (i_lim % block_size)) % block_size),
                (0, (block_size - (j_lim % block_size)) % block_size),
            ],
            mode='constant'
        )
    else:
        img = np.pad(
            img,
            [
                (0, (block_size - (i_lim % block_size)) % block_size),
                (0, (block_size - (j_lim % block_size)) % block_size),
                (0, 0)
            ],
            mode='constant'
        )

    for i in range(0, i_lim, block_size):
        for j in range(0, j_lim, block_size):
            y_slice = img[i:i + block_size, j:j + block_size]
            blocks.append(y_slice)

    return np.array(blocks)


def reconstruct_matrix(blocks: np.ndarray, block_size: int) -> np.ndarray:

    nof_blocks = len(blocks)
    nof_row_blocks = int(np.sqrt(nof_blocks))

    if nof_row_blocks ** 2 != nof_blocks:
        raise ValueError("The number of blocks must be a perfect square")

    original_matrix = np.zeros((nof_row_blocks * block_size, nof_row_blocks * block_size, 3), dtype=int)

    for i, block in enumerate(blocks):
        row_idx = (i // nof_row_blocks) * block_size
        col_idx = (i % nof_row_blocks) * block_size
        original_matrix[row_idx:row_idx + block_size, col_idx:col_idx + block_size] = block

    return original_matrix