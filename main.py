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

figures_dir = './figures'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    host_img = imageio.imread('lena.png')

    embedded_img = scipy.misc.face()
    embedded_img = cv2.cvtColor(embedded_img, cv2.COLOR_RGB2GRAY)
    ret, bw_img = cv2.threshold(embedded_img, 127, 255, cv2.THRESH_BINARY)

    enc = en.arnolds_cat_map_scramble(host_img, 1)

    plt.imshow(enc)
    plt.savefig(f"./{figures_dir}/scrambled_components.pdf")
    plt.clf()

    enc = en.arnolds_cat_map_scramble_inverse(enc, 1)

