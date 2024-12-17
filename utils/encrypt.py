import matplotlib.pyplot as plt
import numpy as np


def arnolds_cat_map_scramble(image_array: np.ndarray, key: int=1):
    n, m = image_array.shape[:2]

    try:
        is_gray = image_array.shape[2] == 1
    except IndexError:
        is_gray = 1

    ret = image_array.copy()

    if is_gray:
        ret = np.pad(
            ret,
            [
                ((np.max([n, m]) - n) // 2, (np.max([n, m]) - n + 1) // 2),  # Padding for height
                ((np.max([n, m]) - m) // 2, (np.max([n, m]) - m + 1) // 2),  # Padding for width
            ],
            mode='constant'
        )
    else:
        ret = np.pad(
            ret,
            [
                ((np.max([n, m]) - n) // 2, (np.max([n, m]) - n + 1) // 2),  # Padding for height
                ((np.max([n, m]) - m) // 2, (np.max([n, m]) - m + 1) // 2),  # Padding for width
                (0, 0)
            ],
            mode="constant",
        )

    n = np.max([n, m])
    for _ in range(key):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(n):
                new_image[(x + y) % n, (x + 2 * y) % n] = ret[x, y]
        ret = new_image

    return ret


def arnolds_cat_map_scramble_inverse(image_array: np.ndarray, key: int=1):
    # design for square images
    n = image_array.shape[0]
    ret = image_array.copy()

    for _ in range(key):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(n):
                new_image[(2 * x - y) % n, (-x + y) % n] = ret[x, y]
        ret = new_image

    return ret