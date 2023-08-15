import cv2
import numpy as np
import scipy

def GetDCTSingleSegment(maskedImage, useGrayscale):
    # Grayscale DCT
    if useGrayscale:
        gray_scale = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
        dct = scipy.fftpack.dct( scipy.fftpack.dct(gray_scale, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    # RGB DCT
    else:
        dct = scipy.fftpack.dct( scipy.fftpack.dct(maskedImage, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    magnitude_spectrum = np.log1p(np.abs(dct))
    return magnitude_spectrum