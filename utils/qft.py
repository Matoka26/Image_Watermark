import quaternion
import numpy as np
from utils import conversions as conv
from utils import displays as dp


def qft(signal: np.ndarray, visualize_spectrum: bool=False) -> quaternion:
    signal = conv.rgb_to_quat(signal)
    signal = quaternion.as_float_array(signal) # add another 0 dimension
    red = signal[:, :, 1]
    green = signal[:, :, 2]
    blue = signal[:, :, 3]
    red_ft = np.fft.fft2(red)
    green_ft = np.fft.fft2(green)
    blue_ft = np.fft.fft2(blue)

    if visualize_spectrum:
        dp.visualize_qft_spectrum(red_ft, green_ft, blue_ft)

    mu = np.quaternion(0, 0, 1, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = i*(np.real(red_ft) + mu*np.imag(red_ft)) + \
         j*(np.real(green_ft) + mu*np.imag(green_ft)) + \
         k*(np.real(blue_ft) + mu*np.imag(blue_ft))

    real_comp = np.vectorize(lambda q: q.real)(ft)
    i_comp = np.vectorize(lambda q: q.imag[0])(ft)
    j_comp = np.vectorize(lambda q: q.imag[1])(ft)
    k_comp = np.vectorize(lambda q: q.imag[2])(ft)

    return ft


def iqft(signal: quaternion, visualize_qft_components: bool=False) -> np.ndarray:
    signal = quaternion.as_float_array(signal)
    quat_real = signal[:, :, 0]
    quat_i = signal[:, :, 1]
    quat_j = signal[:, :, 2]
    quat_k = signal[:, :, 3]

    if visualize_qft_components:
        dp.visualize_qft_components(quat_real, quat_i, quat_j, quat_k)

    real_ift = np.fft.ifft2(quat_real)
    i_ift = np.fft.ifft2(quat_i)
    j_ift = np.fft.ifft2(quat_j)
    k_ift = np.fft.ifft2(quat_k)

    mu = np.quaternion(0, 0, 1, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = np.real(real_ift) + mu*np.imag(real_ift) + \
         i*(np.real(i_ift) + mu*np.imag(i_ift)) + \
         j*(np.real(j_ift) + mu*np.imag(j_ift)) + \
         k*(np.real(k_ift) + mu*np.imag(k_ift))


    return ft

