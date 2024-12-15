import os
import imageio
from utils import qft
from utils import conversions as conv
import matplotlib.pyplot as plt
from utils import displays as dp

figures_dir = './figures'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    # img = scipy.misc.face()
    img = imageio.imread('lena.png')
    dp.visualize_color_components(img)

    qft_img = qft.qft(img, visualize_spectrum=True)

    img_spectrum = qft.iqft(qft_img, visualize_qft_components=True)

    new_img = conv.quat_to_rgb(img_spectrum)

    plt.imshow(new_img)
    plt.savefig(f"./{figures_dir}/processed_image.pdf")
    plt.show()

