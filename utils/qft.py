import quaternion
import numpy as np
from utils import conversions as conv
from utils import displays as dp


def qft(signal: np.ndarray) -> quaternion:
    """
    Applies the Fast Quaternion Fourier Transform on a 3 channel signal
    Parameters:
        signal (np.ndarray): 3D signal
    Returns:
        quaternion: The spectrum of the image
    """
    signal = conv.rgb_to_quat(signal)
    signal = quaternion.as_float_array(signal)
    red = signal[:, :, 1]
    green = signal[:, :, 2]
    blue = signal[:, :, 3]
    red_ft = np.fft.fft2(red)
    green_ft = np.fft.fft2(green)
    blue_ft = np.fft.fft2(blue)

    mu = np.quaternion(0, 0, 1, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = i*(np.real(red_ft) + mu*np.imag(red_ft)) + \
         j*(np.real(green_ft) + mu*np.imag(green_ft)) + \
         k*(np.real(blue_ft) + mu*np.imag(blue_ft))

    return ft


def iqft(signal: quaternion) -> np.ndarray:
    """
    Computes the Fast Quaternion Fourier Inverse Transform
    Parameters:
        signal(np.ndarray): quaternion-valued array representing the spectrum of a 3D signal
    Returns:
        np.ndarray: real-valued array representing the time domain signal
    """
    signal = quaternion.as_float_array(signal)
    quat_real = signal[:, :, 0]
    quat_i = signal[:, :, 1]
    quat_j = signal[:, :, 2]
    quat_k = signal[:, :, 3]

    real_ift = np.fft.ifft2(quat_real)
    i_ift = np.fft.ifft2(quat_i)
    j_ift = np.fft.ifft2(quat_j)
    k_ift = np.fft.ifft2(quat_k)

    mu = np.quaternion(0, 0, 1, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = np.real(real_ift) + mu*np.imag(real_ift) + \
         i * (np.real(i_ift) + mu*np.imag(i_ift)) + \
         j * (np.real(j_ift) + mu*np.imag(j_ift)) + \
         k * (np.real(k_ift) + mu*np.imag(k_ift))

    return ft

