import cv2
import numpy as np

def GetFFTSingleSegment(maskedImage, useGrayscale):
    # Grayscale FFT
    if useGrayscale:
        gray_scale = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray_scale)
    # RGB FFT
    else:
        fft = np.fft.fft2(maskedImage, axes=(0, 1))
    magnitude_spectrum = np.abs(fft)
    magnitude_spectrum = np.log(np.fft.fftshift(magnitude_spectrum) + 1)
    return magnitude_spectrum