import cv2
import math
import numpy as np
from skimage.transform import match_histograms
from utils import display_img, display_cumulative_histograms, pyramid_g, display_histogram
from image_aligment import alignImages
from exposure_fusion.image import Image, show
from scipy import ndimage, misc
from exposure_fusion.HDR_exposure_fusion import LaplacianMap
from typing import Tuple, Any


def threshold_color(h: np.ndarray, bins: np.ndarray) -> float:
    """
    :param h:       hitogram
    :param bins:    bins
    :return:        Threshold color
    """
    Tc_list = []
    for i in range(0, len(bins)-2):
        NpTc = h[i]
        Np_Tc_ = h[i+1]
        Tc_list.append(abs(NpTc - Np_Tc_))
    Tc_list = np.array(Tc_list)
    index_max = np.where(Tc_list == Tc_list.max())[0][0]
    return (bins[index_max + 1] + bins[index_max])/2.


def bin_map(img_b: np.ndarray, img_g: np.ndarray, img_r: np.ndarray, tc_b: float, tc_g: float, tc_r:float) -> np.ndarray:
    """
    :param img_b:   Image when min=0 and max=1, chanel Blue
    :param img_g:   Image when min=0 and max=1, chanel Green
    :param img_r:   Image when min=0 and max=1, chanel Red
    :param tc_b:    Threshold color blue
    :param tc_g:    Threshold color green
    :param tc_r:    Threshold color red
    :return:        binary Map
    """
    h, w = img_b.shape
    binM = np.full(img_b.shape, 0, dtype=img_b.dtype)
    for y in range(h):
        for x in range(w):
            if img_b[y,x] > tc_b or img_g[y,x] > tc_g or img_r[y,x] > tc_r:
                binM[y,x] = 1
    # display_img(binM, "Bin Map - Normal")
    return binM


def get_bin_maps(imgb_name: str, imgd_name: str) -> Tuple[Any, Any]:
    """
    :param imgb_name:
    :param imgd_name:
    :return:
    """
    I_b = cv2.imread(imgb_name)  # bright Image
    I_d = cv2.imread(imgd_name)  # dark Image

    # I_d is a source image, using Histogram Matching(HM)
    I_h = match_histograms(I_d, reference=I_b, multichannel=True)  # HM Image

    # Display base images into same window
    numpy_h = np.hstack((I_d, I_b, I_h))
    # numpy_h_concat = np.concatenate((I_d, I_b), axis=1)  # another form
    title_img = '(I_d, I_b, I_h) Images'
    display_img(numpy_h, title_img, (300 * 3, 300))

    # Feature detection
    I_w, _ = alignImages(I_b, reference=I_h, algorithm="ORB")  # motion registration

    # Display base images into same window
    numpy_h = np.hstack((I_d, I_b, I_h, I_w))
    title_img = '(I_d, I_b, I_h, I_w) Images'
    display_img(numpy_h, title_img, (300 * 4, 300))

    # WARNING: Use display_img() before of display_cumulative_histograms() generate a error  (ERROR LIBRARY libc++aby)
    # Display cumulative histograms of Histogram Matching(HM)
    # display_cumulative_histograms(I_d, I_b, I_h)

    # we use a Gaussian Pyramid which acts as a low-pass filter
    pyramid_Iw = pyramid_g(I_w, dowsample=5)
    pyramid_Ih = pyramid_g(I_h, dowsample=5)

    # The difference image is computed
    I_diff_g = []
    len_pyramid = len(pyramid_Iw)
    for i in range(len_pyramid):
        I_diff_g.append(cv2.absdiff(pyramid_Iw[i], pyramid_Ih[i]))

    # gaussian expanded
    I_diff_exp = [I_diff_g[len_pyramid - 1]]
    for i in range(len_pyramid - 1, 0, -1):
        size = (I_diff_g[i - 1].shape[1], I_diff_g[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(I_diff_g[i], dstsize=size)
        I_diff_exp.append(gaussian_expanded)

    # Reconstructed gaussian
    I_diff = I_diff_exp[0]
    for i in range(1, len_pyramid):
        size = (I_diff_exp[i].shape[1], I_diff_exp[i].shape[0])
        I_diff = cv2.pyrUp(I_diff, dstsize=size)
        I_diff = cv2.add(I_diff_exp[i], I_diff)
    # display_img(I_diff, title="reconstrcted")

    # The goal of the next steps is to accurately distinguish between
    # the previously mentioned difference values. To this end, we
    # process I_diff using the logistic function
    k1, k2 = 0.09, 12
    I_diff_B = np.zeros((I_diff.shape[0], I_diff.shape[1]), dtype=I_diff.dtype)
    I_diff_G = np.zeros((I_diff.shape[0], I_diff.shape[1]), dtype=I_diff.dtype)
    I_diff_R = np.zeros((I_diff.shape[0], I_diff.shape[1]), dtype=I_diff.dtype)
    print(I_diff.shape, I_diff_B.shape)
    for i in range(I_diff.shape[1]):
        for j in range(I_diff.shape[0]):
            I_diff_B[j, i] = 1. / (1. + k1 * math.exp(-k2 * (I_diff[j, i][0] - 0.5)))
            I_diff_G[j, i] = 1. / (1. + k1 * math.exp(-k2 * (I_diff[j, i][1] - 0.5)))
            I_diff_R[j, i] = 1. / (1. + k1 * math.exp(-k2 * (I_diff[j, i][2] - 0.5)))

    numpy_bgr = np.hstack((I_diff_B, I_diff_G, I_diff_R))
    display_img(numpy_bgr, title="logistic function: (I_diff_B, I_diff_G, I_diff_R)", resize=(300 * 3, 300))

    # Calculamos los histogramas
    histogram_IdiffB, bin_edges_B = np.histogram(I_diff_B.ravel(), bins='auto')
    histogram_IdiffG, bin_edges_G = np.histogram(I_diff_G.ravel(), bins='auto')
    histogram_IdiffR, bin_edges_R = np.histogram(I_diff_R.ravel(), bins='auto')

    Tc_B = threshold_color(histogram_IdiffB, bin_edges_B)
    Tc_G = threshold_color(histogram_IdiffG, bin_edges_G)
    Tc_R = threshold_color(histogram_IdiffR, bin_edges_R)

    M = bin_map(I_diff_B, I_diff_G, I_diff_R, Tc_B, Tc_G, Tc_R)
    print("RAYOSSS:",M.dtype)
    # These operations aim at removing possible detection noise (wrongly detected pixels)
    # and enhance the shape and ﬁlling of motion objects in the ﬁnal motion map.
    kernel = np.zeros((5, 5), np.uint8)     # normalmente es un kernel de unos
    M_ = cv2.erode(M, kernel, iterations=1)
    M_ = cv2.dilate(M, kernel, iterations=1)
    print(M.shape, M.dtype)

    # the motion map corresponding to the designated reference image is composed of ones,
    # as we assume that all pixels in the reference image are valid.
    shape = (I_b.shape[0], I_b.shape[1])
    bin_map_ref_img = np.full(shape, 1)

    M_ = M
    display_img(M_, title="M - erode_dilate")

    return bin_map_ref_img, M_


def execute(imgb_name: str, imgd_name: str) -> None:
    """
    :param imgb_name:
    :param imgd_name:
    :return:
    """
    bM_ref, bM = get_bin_maps(imgb_name, imgd_name)

    # HDR De-ghosting --------------------------------

    names = [line.rstrip('\n') for line in open('list_images.txt')]
    #lap = LaplacianMap('tren', names, [bM, bM_ref], n=6)
    lap = LaplacianMap('tren', names, [bM, bM], n=6)
    res = lap.result_exposure(1, 1, 1)
    show(res)
    misc.imsave("res/arno_6.jpg", res)
    return None

    


