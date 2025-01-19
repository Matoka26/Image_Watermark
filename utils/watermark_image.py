import numpy as np
import quaternion
from utils import conversions as conv
from utils import encrypt as en
from utils import qft


def watermark_iamge(host: np.ndarray, watermark: np.ndarray, embedding_strenght: int=2) -> np.ndarray:
    scrambled_watermark = en.arnolds_cat_map_scramble(watermark, 1)

    host_blocks = conv.break_image_to_blocks(host, 8)
    wm_blocks = conv.break_image_to_blocks(scrambled_watermark, 2)

    real_part_blocks = np.array([np.vectorize(lambda q: q.real)(qft.qft(block)) for block in host_blocks])

    new_real_part_blocks = en.embed_watermark(real_part_blocks, wm_blocks, 1)

    embeded_host = []
    for i, block in enumerate(new_real_part_blocks):

        qft_block = qft.qft(host_blocks[i])
        if i == 30:
            print('initial')
            print(qft_block[1])

        for k in range(len(qft_block)):
            for l in range(len(qft_block[k])):
                qft_block[k, l] = np.quaternion(
                    new_real_part_blocks[i][k, l],
                    qft_block[k, l].x,
                    qft_block[k, l].y,
                    qft_block[k, l].z
                )

        if i == 30:
            print('after embed')
            print(qft_block[1])
            rec = qft.iqft(qft_block)
            rec = conv.quat_to_rgb(rec)
            print('new qft')
            print(qft.qft(rec)[1])

        embeded_host.append(conv.quat_to_rgb(qft.iqft(qft_block)))

    embeded_host = np.array(embeded_host)
    embeded_host = conv.reconstruct_matrix(embeded_host, 8)

    return embeded_host