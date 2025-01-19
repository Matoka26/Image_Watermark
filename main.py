import copy
import os
import imageio
from utils import qft
from utils import conversions as conv
import matplotlib.pyplot as plt
from utils import displays as dp
import scipy
from utils import encrypt as en
import numpy as np
import cv2
from utils import watermark_image as wi
import skimage

import mahotas

figures_dir = './figures'
outputs_dir = './outputs'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)

    host_img = cv2.imread('hosts/lena.png', cv2.IMREAD_COLOR)
    host_img = cv2.resize(host_img, (800, 800), interpolation=cv2.INTER_AREA)
    watermark = cv2.imread('watermarks/catface.jpg', cv2.IMREAD_GRAYSCALE)
    new_image = wi.watermark_iamge(host_img, watermark)
    # new_image = wi.watermark_iamge(new_image, watermark)
    new_image = np.uint8(new_image)
    cv2.imshow('cox', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # host_blocks = conv.break_image_to_blocks(host, 8)
    # real_part_blocks = np.array([np.vectorize(lambda q: q.real)(qft.qft(block)) for block in host_blocks])
    #
    # for i in range(len(real_part_blocks)):
    #     # daca alegi mai mic de asta original, te arunca la un trashold, daca pui mai mare te arunca destul in sus
    #     real_part_blocks[i][1, 1] = int(real_part_blocks[i][1, 1]) +1
    #     real_part_blocks[i][7, 7] = - real_part_blocks[i][1, 1]
    #
    # embeded_host = []
    # for i, block in enumerate(real_part_blocks):
    #
    #     qft_block = qft.qft(host_blocks[i])
    #
    #     if i == 100:
    #         print(qft_block[1][1].real)
    #     for k in range(len(qft_block)):
    #         for l in range(len(qft_block[k])):
    #             qft_block[k, l] = np.quaternion(
    #                 real_part_blocks[i][k, l],
    #                 qft_block[k, l].x,
    #                 qft_block[k, l].y,
    #                 qft_block[k, l].z
    #             )
    #
    #     if i == 100:
    #         print(qft_block[1][1].real)
    #
    #         rec = conv.quat_to_rgb(qft.iqft(qft_block))
    #         print(qft.qft(rec)[1][1].real)
    #
    #     embeded_host.append(conv.quat_to_rgb(qft.iqft(qft_block)))
    #
    # embeded_host = np.array(embeded_host)
    # embeded_host = conv.reconstruct_matrix(embeded_host, 8)
    # aici e iar ok

    # plt.imshow(embeded_host)
    # plt.show()
