import copy

import matplotlib.pyplot as plt
import numpy as np
import warnings


def arnolds_cat_map_scramble(image_array: np.ndarray, key: int=1) -> np.ndarray:
    n = image_array.shape[0]
    if n != image_array.shape[1]:
        warnings.warn('Input image is not square')

    ret = copy.deepcopy(image_array)
    for _ in range(key):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(n):
                new_image[(x + y) % n, (x + 2 * y) % n] = ret[x, y]
        ret = new_image

    return ret


def arnolds_cat_map_scramble_inverse(image_array: np.ndarray, key: int=1) -> np.ndarray:
    n = image_array.shape[0]
    if n != image_array.shape[1]:
        warnings.warn('Input image is not square')

    ret = image_array.copy()

    for _ in range(key):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(n):
                new_image[(2 * x - y) % n, (-x + y) % n] = ret[x, y]
        ret = new_image

    return ret


def find_cat_map_key(image: np.ndarray) -> int:
    new_img = copy.deepcopy(image)
    new_img = arnolds_cat_map_scramble(new_img)

    i = 1
    while not (new_img == image).all():
        new_img = arnolds_cat_map_scramble(new_img)
        i += 1

    return i

def embed_watermark(host_blocks: np.ndarray, watermark_blocks: np.ndarray, embedding_strenght: int=1) -> np.ndarray:

    embedding_strenght *= 2
    for i in range(watermark_blocks.shape[0]):
        host_blocks[i][1, 1] = (-1)**(watermark_blocks[i][0, 0] == 0) * 20

        host_blocks[i][1, 2] = (-1)**(watermark_blocks[i][0, 1] == 0) * 20

        host_blocks[i][2, 1] = (-1)**(watermark_blocks[i][1, 0] == 0) * 20

        host_blocks[i][2, 2] = (-1)**(watermark_blocks[i][1, 1] == 0) * 20

        host_blocks[i][6, 6] = - host_blocks[i][2, 2]
        host_blocks[i][6, 7] = - host_blocks[i][2, 1]
        host_blocks[i][7, 6] = - host_blocks[i][1, 2]
        host_blocks[i][7, 7] = - host_blocks[i][1, 1]

    return host_blocks