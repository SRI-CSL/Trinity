import cv2
import numpy as np
import scipy

def GetEntropySingleSegment(maskedImage, useGrayscale):
    # Grayscale Entropy
    if useGrayscale:
        gray_scale = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
        entropy = scipy.entropy( scipy.entropy(gray_scale, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    # RGB Entropy
    else:
        entropy = scipy.entropy( scipy.entropy(gray_scale, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    magnitude_spectrum = np.log1p(entropy)
    return magnitude_spectrum