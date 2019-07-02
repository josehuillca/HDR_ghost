import cv2
import math
import numpy as np
from skimage.transform import match_histograms
from utils import display_img, display_cumulative_histograms, pyramid_g, display_histogram
from image_aligment import alignImages


def execute(imgb_name: str, imgd_name: str) -> None:
    """
    :param imgb_name:
    :param imgd_name:
    :return:
    """
    I_b = cv2.imread(imgb_name)     # bright Image
    I_d = cv2.imread(imgd_name)     # dark Image

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
            I_diff_B[j, i] = 1./(1. + k1*math.exp(-k2*(I_diff[j, i][0] - 0.5)))
            I_diff_G[j, i] = 1./(1. + k1*math.exp(-k2*(I_diff[j, i][1] - 0.5)))
            I_diff_R[j, i] = 1./(1. + k1*math.exp(-k2*(I_diff[j, i][2] - 0.5)))

    numpy_bgr = np.hstack((I_diff_B, I_diff_G, I_diff_R))
    display_img(numpy_bgr, title="logistic function: (I_diff_B, I_diff_G, I_diff_R)", resize=(300 * 3, 300))

    # Calculamos los histogramas
    histogram_IdiffB, bin_edges_B = np.histogram(I_diff_B.ravel(), bins='auto')
    histogram_IdiffG, bin_edges_G = np.histogram(I_diff_G.ravel(), bins='auto')
    histogram_IdiffR, bin_edges_R = np.histogram(I_diff_R.ravel(), bins='auto')
    print(I_diff_B)
    print("Bin edges:", bin_edges_B[:20])
    print(histogram_IdiffB.shape, histogram_IdiffG.shape, histogram_IdiffR.shape)
    print(bin_edges_B.shape, bin_edges_G.shape, bin_edges_R.shape)

    Tc_B = threshold_color(histogram_IdiffB, bin_edges_B)
    Tc_G = threshold_color(histogram_IdiffG, bin_edges_G)
    Tc_R = threshold_color(histogram_IdiffR, bin_edges_R)
    print(Tc_B, Tc_G, Tc_R, max(bin_edges_B))
    #display_histogram(I_diff_B)
    bin_map(I_diff_B, Tc_B)
    bin_map(I_diff_G, Tc_G)
    bin_map(I_diff_R, Tc_R)


def threshold_color(h: np.ndarray, bins: np.ndarray) -> float:
    """
    :param h:       hitogram
    :param bins:    bins
    :return:        argmax of Tc_list
    """
    Tc_list = []
    for i in range(0, len(bins)-2):
        NpTc = h[i]
        Np_Tc_ = h[i+1]
        Tc_list.append(abs(NpTc - Np_Tc_))
    Tc_list = np.array(Tc_list)
    index_max = np.where(Tc_list == Tc_list.max())[0][0]
    print(np.where(Tc_list == Tc_list.max()))
    return (bins[index_max + 1] + bins[index_max])/2.


def bin_map(img, tc):
    h, w = img.shape
    binM = np.full(img.shape, 1, dtype=img.dtype)
    for y in range(h):
        for x in range(w):
            if img[y,x] > tc:
                binM[y,x] = 0
    display_img(binM, "Bin Map")