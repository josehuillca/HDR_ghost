import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.transform import match_histograms
from utils import display_img, display_cumulative_histograms
from image_aligment import alignImages


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
    I_w, h = alignImages(I_b, reference=I_h, algorithm="SIFT")  # motion registration
    
    # Display base images into same window
    numpy_h = np.hstack((I_d, I_b, I_h, I_w))
    title_img = '(I_d, I_b, I_h, I_w) Images'
    display_img(numpy_h, title_img, (300*4, 300))

    # WARNING: Use display_img() before of display_cumulative_histograms() generate a error  (ERROR LIBRARY libc++aby)
    # Display cumulative histograms of Histogram Matching(HM)
    # display_cumulative_histograms(I_d, I_b, I_h)

    print("Finished...!")
