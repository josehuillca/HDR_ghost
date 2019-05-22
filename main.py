import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any
from skimage import exposure
from skimage.transform import match_histograms


def display_img(img: np.ndarray, title: str, resize: np.ndarray = (600, 600)) -> None:
    """ Display image window
    :param img: Input image
    :param title: Title image
    :param resize: Resize window (width, height)
    :return:
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


def feature_detection(image: np.ndarray, algorithm: str = 'SIFT') -> Tuple[Any, Any]:
    """ USE SIFT ALGORITHM or SURF to extract features
    :param image: Input image
    :param algorithm: SIFT or SURF
    :return: (keypoints, descriptors)
    """
    image_cp = copy.copy(image)
    image_cp = cv2.cvtColor(image_cp, cv2.COLOR_BGR2GRAY)
    if algorithm == 'SIFT':
        alg = cv2.xfeatures2d.SIFT_create()
    else:
        alg = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = alg.detectAndCompute(image_cp, None)
    return keypoints, descriptors


if __name__ == "__main__":
    I_b = cv2.imread("images/cup_bright.png")  # bright Image
    I_d = cv2.imread("images/cup_dark.png")  # dark Image

    # I_d is a source image, using Histogram Matching(HM)
    I_h = match_histograms(I_d, reference=I_b, multichannel=True)  # HM Image

    # Display base images into same window
    numpy_h = np.hstack((I_d, I_b, I_h))
    # numpy_h_concat = np.concatenate((I_d, I_b), axis=1)  # another form
    title_img = '(I_d, I_b, I_h) Images'
    display_img(numpy_h, title_img, (300*3, 300))

    # Feature detection
    key_p_Ib, _ = feature_detection(I_b, algorithm='SIFT')
    key_p_Ih, _ = feature_detection(I_h, algorithm='SIFT')

    # Display images with keypoints
    img_key_p_Ib = cv2.drawKeypoints(I_b, key_p_Ib, None)
    img_key_p_Ih = cv2.drawKeypoints(I_h, key_p_Ih, None)
    numpy_h_key_p = np.hstack((img_key_p_Ib, img_key_p_Ih))
    title_img_key_p = '(img_key_p_Ib, img_key_p_Ih) Images'
    display_img(numpy_h_key_p, title_img_key_p, (600 * 2, 600))

    # WARNING: Use display_img() before of display_cumulative_histograms() generate a error  (ERROR LIBRARY libc++aby)
    # Display cumulative histograms of Histogram Matching(HM)
    # display_cumulative_histograms(I_d, I_b, I_h)

    print("Finished...!")
