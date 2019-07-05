import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from exposure_fusion.image import *
from exposure_fusion.utils import *
import pdb

#
#def div0( a, b ):
#    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
#    with np.errstate(divide='ignore', invalid='ignore'):
#        c = np.true_divide( a, b )
#        c[ ~ np.isfinite( c )] = 0
#        return c


class LaplacianMap(object):
    """Class for weights attribution with Laplacian Fusion"""

    def __init__(self, fmt, names, bin_maps, n=3):
        """names is a liste of names, fmt is the format of the images"""
        self.images = []
        self.bin_maps = bin_maps
        for name in names:
            self.images.append(Image(fmt, name, crop=True, n=n))
        self.shape = self.images[0].shape
        self.num_images = len(self.images)
        self.height_pyr = n

    def get_weights_map(self, w_c: float, w_s: float, w_e: float):
        """Return the normalized Weight map"""
        self.weights = []
        sums = np.zeros((self.shape[0], self.shape[1]))
        i = 0
        for image_name in self.images:
            contrast = image_name.contrast()
            saturation = image_name.saturation()
            exposedness = image_name.exposedness()

            # Usando Laplacian y pyramid gaussian se pierden unos pixeles
            print(contrast.shape, (self.shape[0], self.shape[1]), np.array(self.bin_maps[i]).shape)
            new_bin_map = misc.imresize(np.array(self.bin_maps[i]), (self.shape[0], self.shape[1]))
            w_add = 1e-12
            if i == -1:
                w_add = 0
            weight = (contrast ** w_c) * (saturation ** w_s) * (exposedness ** w_e) * new_bin_map + w_add
            self.weights.append(weight)
            sums = sums + weight
            i = i + 1
        for index in range(self.num_images):
            self.weights[index] = self.weights[index] / sums
        return self.weights

    def get_gaussian_pyramid(self, image, n):
        """Return the Gaussian Pyramid of an image"""
        print("get_gaussian_pyramid")
        gaussian_pyramid_floors = [image]
        for floor in range(1, n):
            gaussian_pyramid_floors.append(
                Reduce(gaussian_pyramid_floors[-1], 1))
        return gaussian_pyramid_floors

    def get_gaussian_pyramid_weights(self):
        """Return the Gaussian Pyramid of the Weight map of all images"""
        self.weights_pyramid = []
        for index in range(self.num_images):
            self.weights_pyramid.append(
                self.get_gaussian_pyramid(self.weights[index],
                                          self.height_pyr))
        return self.weights_pyramid

    def get_laplacian_pyramid(self, image, n):
        """Return the Laplacian Pyramid of an image"""
        print("get_laplacian_pyramid")
        gaussian_pyramid_floors = self.get_gaussian_pyramid(image, n)
        laplacian_pyramid_floors = [gaussian_pyramid_floors[-1]]
        for floor in range(n - 2, -1, -1):
            print(floor)
            new_floor = gaussian_pyramid_floors[floor] - Expand(
                gaussian_pyramid_floors[floor + 1], 1)
            laplacian_pyramid_floors = [new_floor] + laplacian_pyramid_floors
        return laplacian_pyramid_floors

    def get_laplacian_pyramid_images(self):
        """Return all the Laplacian pyramid for all images"""
        self.laplacian_pyramid = []
        for index in range(self.num_images):
            self.laplacian_pyramid.append(
                self.get_laplacian_pyramid(self.images[index].array,
                                           self.height_pyr))
        return self.laplacian_pyramid

    def result_exposure(self, w_c=1, w_s=1, w_e=1):
        "Return the Exposure Fusion image with Laplacian/Gaussian Fusion method"
        print("weights")
        self.get_weights_map(w_c, w_s, w_e)
        print("gaussian pyramid")
        self.get_gaussian_pyramid_weights()
        print("laplacian pyramid")
        self.get_laplacian_pyramid_images()
        result_pyramid = []
        print("PASOOOO---------")
        for floor in range(self.height_pyr):
            print('floor ', floor)
            result_floor = np.zeros(self.laplacian_pyramid[0][floor].shape)
            for index in range(self.num_images):
                print('floor ', floor)
                for canal in range(3):
                    result_floor[:, :,
                                 canal] += self.laplacian_pyramid[index][floor][:, :,
                                                                                canal] * self.weights_pyramid[index][floor]
            result_pyramid.append(result_floor)
        # Get the image from the Laplacian pyramid
        self.result_image = result_pyramid[-1]
        for floor in range(self.height_pyr - 2, -1, -1):
            print('floor ', floor)
            self.result_image = result_pyramid[floor] + Expand(
                self.result_image, 1)
        self.result_image[self.result_image < 0] = 0
        self.result_image[self.result_image > 1] = 1
        return self.result_image
