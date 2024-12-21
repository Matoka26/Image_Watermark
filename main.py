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

figures_dir = './figures'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    host_img = cv2.imread('lena.png')
    watermark = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)

    new_image = wi.watermark_iamge(host_img, watermark)
    new_image = np.uint8(new_image)

    cv2.imshow('Image', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()