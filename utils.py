import cv2
import numpy as np
import matplotlib.pyplot as plt

from sys import platform as _platform
if _platform == "darwin":
   # MAC OS X
   import matplotlib
   matplotlib.use('MacOSX')

from skimage import exposure
from skimage.transform import pyramid_gaussian
from typing import Tuple, List


def display_img(img: np.ndarray, title: str, resize: Tuple[int, int] = (600, 600)) -> None:
    """ Display image window
    :param img: Input image
    :param title: Title image
    :param resize: Resize window (width, height)
    :return: None
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, resize[0], resize[1])
    cv2.waitKey()
    cv2.destroyWindow(title)


def display_cumulative_histograms(source: np.ndarray, reference: np.ndarray, matched: np.ndarray) -> None:
    """ Display cumulative and histograms of images RGB, used to the function matched_histograms()
    :param source: image source
    :param reference: image reference
    :param matched: image result of matched_histograms()
    :return: None
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

    for i, img in enumerate((source, reference, matched)):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[..., c])
            axes[c, i].plot(bins, img_cdf)
            axes[c, 0].set_ylabel(c_color)

    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')

    plt.tight_layout()
    plt.show()


def pyramid_g(img: np.ndarray, dowsample: int = 2, downscale: int = 2) -> List:
    gaussian_pyramid = []
    # METHOD #2: Resizing + Gaussian smoothing.
    for (i, resized) in enumerate(pyramid_gaussian(img, downscale=downscale, multichannel=True)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        if i > dowsample:
            break
        # show the resized image
        gaussian_pyramid.append(resized)
    return gaussian_pyramid


def display_histogram(img: np.ndarray) -> None:
    """
    :param img:
    :param min_v:
    :param max_v:
    :return:
    """
    plt.hist(img.ravel(), bins='auto')
    plt.show()