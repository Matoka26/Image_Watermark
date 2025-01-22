from utils import watermark_image as wi
from utils import encrypt as en
import matplotlib.pyplot as plt
import os
import cv2


figures_dir = './figures'
outputs_dir = './outputs'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)

    host_img = cv2.imread('hosts/lena.png', cv2.IMREAD_COLOR)
    host_img = cv2.resize(host_img, (800, 800), interpolation=cv2.INTER_AREA)

    watermark = cv2.imread('watermarks/football.jpg', cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (200, 200), interpolation=cv2.INTER_AREA)
    _, watermark = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)

    new_image = wi.watermark_iamge(host_img, watermark, scramble_key=1)
    excracted_watermark = wi.extract_watermark(new_image, watermark_side=watermark.shape[0], scramble_key=en.find_cat_map_key(watermark)-1)

    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax[0].imshow(watermark)
    ax[0].set_title('Watermark')
    ax[1].imshow(excracted_watermark)
    ax[1].set_title('Extracted Watermark')
    plt.show()