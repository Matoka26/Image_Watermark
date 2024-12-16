import numpy as np


def arnolds_cat_map_scramble(image_array: np.ndarray, iterations: int=1):
    n, m = image_array.shape[:2]
    ret = image_array.copy()

    for _ in range(iterations):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(m):
                new_image[(x + y) % n, (x + 2 * y) % m] = ret[x, y]
        ret = new_image

    return ret


def arnolds_cat_map_scramble_inverse(image_array: np.ndarray, iterations: int=1):
    n, m = image_array.shape[:2]
    ret = image_array.copy()

    for _ in range(iterations):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(m):
                new_image[(2 * x - y) % n, (-x + y) % m] = ret[x, y]
        ret = new_image

    return ret