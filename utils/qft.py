import quaternion
import numpy as np
from utils import conversions as conv
from utils import encrypt as en
import matplotlib.pyplot as plt


def qft(signal: np.ndarray, visualize_spectrum: bool=False) -> quaternion:
    signal = conv.rgb_to_quat(signal)
    signal = quaternion.as_float_array(signal) # add another 0 dimension
    red = signal[:, :, 1]
    green = signal[:, :, 2]
    blue = signal[:, :, 3]
    red_ft = np.fft.fft2(red)
    green_ft = np.fft.fft2(green)
    blue_ft = np.fft.fft2(blue)

    en.hi(red_ft)

    if visualize_spectrum:
        visualize_qft_spectrum(red_ft, green_ft, blue_ft)

    mu = np.quaternion(0, 1, 0, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = i*(np.real(red_ft) + mu*np.imag(red_ft)) + \
         j*(np.real(green_ft) + mu*np.imag(green_ft)) + \
         k*(np.real(blue_ft) + mu*np.imag(blue_ft))

    return ft


def iqft(signal: quaternion) -> np.ndarray:
    signal = quaternion.as_float_array(signal)
    quat_real = signal[:, :, 0]
    quat_i = signal[:, :, 1]
    quat_j = signal[:, :, 2]
    quat_k = signal[:, :, 3]
    real_ift = np.fft.ifft2(quat_real)
    i_ift = np.fft.ifft2(quat_i)
    j_ift = np.fft.ifft2(quat_j)
    k_ift = np.fft.ifft2(quat_k)

    mu = np.quaternion(0, 1, 0, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = np.real(real_ift) + mu*np.imag(real_ift) + \
         i*(np.real(i_ift) + mu*np.imag(i_ift)) + \
         j*(np.real(j_ift) + mu*np.imag(j_ift)) + \
         k*(np.real(k_ift) + mu*np.imag(k_ift))

    return ft


def visualize_qft_spectrum(red_ft: quaternion, green_ft: quaternion, blue_ft: quaternion) -> None:
    # Compute the magnitude spectrum for each component
    red_magnitude = np.abs(np.fft.fftshift(red_ft))
    green_magnitude = np.abs(np.fft.fftshift(green_ft))
    blue_magnitude = np.abs(np.fft.fftshift(blue_ft))

    # Visualize the spectra
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(10 * np.log(1 + red_magnitude), cmap="inferno")
    plt.title("Red Channel Spectrum")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(10 * np.log(1 + green_magnitude), cmap="gist_earth")
    plt.title("Green Channel Spectrum")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(10 * np.log(1 + blue_magnitude), cmap='ocean')
    plt.title("Blue Channel Spectrum")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # remove later
    plt.savefig(f"./figures/color_spectrums.pdf")


