import os
import numpy as np
import matplotlib.pyplot as plt
import quaternion
import scipy.misc
from utils import qft
from utils import conversions as conv

figures_dir = './figures'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    img = scipy.misc.face()

    qft_img = qft.qft(img, visualize_spectrum=True)

    img_spectrum = qft.iqft(qft_img)

    new_img = conv.quat_to_rgb(img_spectrum)

    plt.imshow(new_img)
    plt.savefig(f"./{figures_dir}/processed_image.pdf")
    plt.show()

