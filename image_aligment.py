from __future__ import print_function
import cv2
import copy
import numpy as np
from utils import display_img
from typing import Tuple, Any
 
 
MAX_FEATURES = 500          # parameter to ORB algorithm
GOOD_MATCH_PERCENT = 0.15   # porcentaje de los buenos matches a tomar en cuenta


def feature_detection(image: np.ndarray, algorithm: str = 'SIFT') -> Tuple[Any, Any]:
    """ USE SIFT, SURF or ORB ALGORITHM to extract features
    :param image: Input image BGR
    :param algorithm: SIFT, SURF or ORB string
    :return: (keypoints, descriptors)
    """
    image_cp = copy.copy(image)
    image_cp = cv2.cvtColor(image_cp, cv2.COLOR_BGR2GRAY)
    if algorithm == 'SIFT':
        alg = cv2.xfeatures2d.SIFT_create()
    elif algorithm == 'SURF':
        alg = cv2.xfeatures2d.SURF_create()
    else:
        alg = cv2.cv2.ORB_create(MAX_FEATURES)
    keypoints, descriptors = alg.detectAndCompute(image_cp, None)
    return keypoints, descriptors


def alignImages(source: np.ndarray, reference: np.ndarray, algorithm: str = 'ORB'):

    keypoints1, descriptors1 = feature_detection(source, algorithm)
    keypoints2, descriptors2 = feature_detection(reference, algorithm)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(source, keypoints1, reference, keypoints2, matches, None)
    display_img(imMatches, "Matches", (300*2, 300))
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width, channels = reference.shape
    im1Reg = cv2.warpPerspective(source, h, (width, height))
    
    return im1Reg, h
