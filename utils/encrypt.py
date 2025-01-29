import copy
import numpy as np
import warnings


def arnolds_cat_map_scramble(img_array: np.ndarray, key: int=1) -> np.ndarray:
    """
    Scrambles the pixels of an image according to the transformation:
        F(x, y) = [ 1  1 ] * [ x ]  (mod N), where N is the side length of the image
                  [ 1  2 ]   [ y ]
    Parameters:
        img_array (np.ndarray): A NumPy array representing an image
        key(int): The number of scrambles applied to the image
    Returns:
        np.ndarray: The resulting scrambled image
    """
    n = img_array.shape[0]
    if n != img_array.shape[1]:
        warnings.warn('Input image is not square')

    ret = copy.deepcopy(img_array)
    for _ in range(key):
        new_image = np.zeros_like(ret)
        for x in range(n):
            for y in range(n):
                new_image[(x + y) % n, (x + 2 * y) % n] = ret[x, y]
        ret = new_image

    return ret


def find_cat_map_key(image: np.ndarray) -> int:
    """
    Scrambles the pixels of an image over and over again according to the transformation:
        F(x, y) = [ 1  1 ] * [ x ]  (mod N), where N is the side length of the image
                  [ 1  2 ]   [ y ]
        until the original image reappears, finding the decription key, represented
        by the period of the imaage
    Parameters:
        image (np.ndarray): A NumPy array representing an image
    Returns:
        int: The decription key
    """
    new_img = copy.deepcopy(image)
    new_img = arnolds_cat_map_scramble(new_img)

    i = 1
    while not (new_img == image).all():
        new_img = arnolds_cat_map_scramble(new_img)
        i += 1

    return i


def embed_watermark(host_blocks: np.ndarray, watermark_blocks: np.ndarray) -> np.ndarray:
    """
    Embeds information from a watermark into a host image, both represented as lists of blocks,
    with respect to the following scheme:
        - Each block w(i) from the watermark has a corresponding host block hi
        - elements [1,1], [1,2], [2,1], [2,2] from h(i) will be assigned values 20 or -20
        depending on the value of the value w(i) on positions [0,0], [0,1], [1,0], [1,1]
    Parameters:
        host_blocks (np.ndarray): A list of host blocks
        watermark_blocks (np.ndarray): A list of host blocks
    Returns:
        np.ndarray: A list of embedded blocks
    """

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