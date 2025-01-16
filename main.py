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

    host_img = cv2.imread('hosts/cat.jpg', cv2.IMREAD_COLOR)
    host_img = cv2.resize(host_img, (800, 800), interpolation = cv2.INTER_AREA)
    watermark = cv2.imread('watermarks/catface.jpg', cv2.IMREAD_GRAYSCALE)
    new_image = wi.watermark_iamge(host_img, watermark)
    new_image = np.uint8(new_image)

    cv2.imwrite(f'{outputs_dir}/cat_catface.png', new_image)
