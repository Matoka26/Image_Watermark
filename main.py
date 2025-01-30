from utils import watermark_image as wi
from utils import encrypt as en
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as ssim

def binary_error(image1, image2):
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Count differing pixels
    error = np.sum(image1 != image2)  # Count mismatched pixels
    total_pixels = image1.size  # Total number of pixels
    return error / total_pixels  # Normalize error


def mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")
    return np.mean((image1 - image2) ** 2)


figures_dir = './figures'
outputs_dir = './outputs'

if __name__ == '__main__':
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)

    host_name = 'lena'
    wm_name = 'football'

    host_img = cv2.imread(f'hosts/{host_name}.png', cv2.IMREAD_COLOR)
    host_img = cv2.resize(host_img, (640, 640), interpolation=cv2.INTER_AREA)

    watermark = cv2.imread(f'watermarks/{wm_name}.png', cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (140, 140), interpolation=cv2.INTER_AREA)
    _, watermark = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)

    new_image = wi.watermark_iamge(host_img, watermark, scramble_key=1)
    excracted_watermark = wi.extract_watermark(new_image, watermark_side=watermark.shape[0], scramble_key=en.find_cat_map_key(watermark)-1)

    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(12, 4))

    host_img = cv2.cvtColor(host_img, cv2.COLOR_BGR2RGB)
    ax[0].imshow(host_img)
    ax[0].set_title('Host')
    ax[0].text(0.5, -0.2, f'Shape: {host_img.shape[0]} x {host_img.shape[1]}',
               fontsize=12, ha='center', transform=ax[0].transAxes)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    ax[2].imshow(new_image)
    ax[2].set_title('Embedded Host')
    error_value = mse(host_img, new_image) / host_img.size
    ax[2].text(0.5, -0.2, f'logMSE: {np.log10(error_value + 1e-10) }',
               fontsize=12, ha='center', transform=ax[2].transAxes)

    ax[1].imshow(watermark, cmap="gray")
    ax[1].set_title('Watermark')
    ax[1].text(0.5, -0.2, f'Shape: {watermark.shape[0]} x {watermark.shape[1]}',
               fontsize=12, ha='center', transform=ax[1].transAxes)

    ax[3].imshow(excracted_watermark, cmap="gray")
    ax[3].set_title(f'Extracted Watermark')
    error_value = binary_error(watermark, excracted_watermark)
    ax[3].text(0.5, -0.2, f'Binary Error: {error_value:.4f}',
               fontsize=12, ha='center', transform=ax[3].transAxes)

    plt.tight_layout()
    plt.savefig(f'{outputs_dir}/{host_name}_{wm_name}.png')
    plt.show()

