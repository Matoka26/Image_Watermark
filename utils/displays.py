import matplotlib.pyplot as plt
import quaternion
import numpy as np



def visualize_qft_spectrum(red_ft: quaternion, green_ft: quaternion, blue_ft: quaternion) -> None:
    # Compute the magnitude spectrum for each component
    red_magnitude = np.abs(np.fft.fftshift(red_ft))
    green_magnitude = np.abs(np.fft.fftshift(green_ft))
    blue_magnitude = np.abs(np.fft.fftshift(blue_ft))

    # Visualize the spectra
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(10 * np.log(1 + red_magnitude), cmap="Reds_r")
    plt.title("Red Channel Spectrum")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(10 * np.log(1 + green_magnitude), cmap="Greens_r")
    plt.title("Green Channel Spectrum")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(10 * np.log(1 + blue_magnitude), cmap='Blues_r')
    plt.title("Blue Channel Spectrum")
    plt.colorbar()

    plt.tight_layout()

    # remove later
    plt.savefig(f"./figures/color_spectrums.pdf")

    plt.show()


def visualize_color_components(signal: np.ndarray, cmap=None) -> None:
    red_chan = signal[:, :, 0]
    green_chan = signal[:, :, 1]
    blue_chan = signal[:, :, 2]

    # Visualize the spectra
    plt.figure(figsize=(14, 4))

    r_cmap = "Reds_r"
    g_cmap = "Greens_r"
    b_cmap = "Blues_r"

    if cmap is not None:
        r_cmap = g_cmap = b_cmap = cmap

    plt.subplot(1, 3, 1)
    plt.imshow(red_chan, cmap=r_cmap)
    plt.title("Red Channel")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(green_chan, cmap=g_cmap)
    plt.title("Green Channel")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(blue_chan, cmap=b_cmap)
    plt.title("Blue Channel")
    plt.colorbar()

    plt.tight_layout()
    # remove later
    plt.savefig(f"./figures/color_components.pdf")
    plt.show()


def visualize_qft_components(a: quaternion, b: quaternion, c: quaternion, d: quaternion) -> None:
    # Compute the magnitude spectrum for each component
    real_magnitute = np.abs(np.fft.ifftshift(a))
    red_magnitude = np.abs(np.fft.ifftshift(b))
    green_magnitude = np.abs(np.fft.ifftshift(c))
    blue_magnitude = np.abs(np.fft.ifftshift(d))

    # Visualize the spectra
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(10 * np.log(1 + real_magnitute), cmap="gray")
    plt.title("Real Component")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(10 * np.log(1 + red_magnitude), cmap="gray")
    plt.title("I Component")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.imshow(10 * np.log(1 + green_magnitude), cmap='gray')
    plt.title("J Component")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.imshow(10 * np.log(1 + blue_magnitude), cmap='gray')
    plt.title("K Component")
    plt.colorbar()

    plt.tight_layout()

    # remove later
    plt.savefig(f"./figures/quaternion_components.pdf")

    plt.show()