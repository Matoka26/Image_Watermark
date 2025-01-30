import numpy as np
from utils import conversions as conv
from utils import encrypt as en
from utils import qft


def watermark_iamge(host: np.ndarray, watermark: np.ndarray, scramble_key: int=1) -> np.ndarray:
    """
    Applies an embedding scheme for a square watermark(binary image) over an image's quaternion frequency spectrum.
        The scramble_key is the encryption strength over the watermark image
    Parameters:
        host (np.ndarray): RGB image
        watermark (np.ndarray): Binary image
        scramble_key (int):
    Returns:
        np.ndarray: Embedded RGB image
    """
    scrambled_watermark = en.arnolds_cat_map_scramble(watermark, scramble_key)

    host_blocks = conv.break_image_to_blocks(host, 8)
    wm_blocks = conv.break_image_to_blocks(scrambled_watermark, 2)

    real_part_blocks = np.array([np.vectorize(lambda q: q.real)(qft.qft(block)) for block in host_blocks])

    new_real_part_blocks = en.embed_watermark(real_part_blocks, wm_blocks)

    embeded_host = []
    for i, block in enumerate(new_real_part_blocks):

        qft_block = qft.qft(host_blocks[i])

        for k in range(len(qft_block)):
            for l in range(len(qft_block[k])):
                qft_block[k, l] = np.quaternion(
                    new_real_part_blocks[i][k, l],
                    qft_block[k, l].x,
                    qft_block[k, l].y,
                    qft_block[k, l].z
                )
        embeded_host.append(conv.quat_to_rgb(qft.iqft(qft_block)))

    embeded_host = np.array(embeded_host)
    embeded_host = conv.reconstruct_matrix(embeded_host, 8)

    return np.uint8(embeded_host)


def extract_watermark(host: np.ndarray, watermark_side: int, scramble_key: int=1) -> np.ndarray:
    """
    Assuming the image has an underlying watermark, extracts the binary watermark given it s side length
        and decrypts it with the scramble key
    Parameters:
        host (np.ndarray): RGB image
        watermark_side (int): Side length of the square watermark
        scramble_key (int): Decryption key of the extracted watermark
    Returns:
        np.ndarray: Extracted binary watermark
    """
    host_blocks = conv.break_image_to_blocks(host, 8)[:watermark_side**2 // 4]
    real_part_blocks = np.array([np.vectorize(lambda q: q.real)(qft.qft(block)) for block in host_blocks])

    watermark_blocks = []
    for block in real_part_blocks:
        extracted_block = [[int(block[1,1].real > 0), int(block[1,2].real > 0)],
                           [int(block[1,2].real > 0), int(block[2,2].real > 0)]]

        watermark_blocks.append(extracted_block)
    watermark_blocks = np.array(watermark_blocks)
    watermark = conv.reconstruct_matrix(watermark_blocks, 2, is_binary=True)

    return en.arnolds_cat_map_scramble(watermark, scramble_key)
