def CheckPatch(dct=None, fft=None, entropy=None):
    return False

def GetSingleImagePatch(numSegments, dctFeatures=None, fftFeatures=None, entropyFeatures=None):
    if entropyFeatures is not None:
        for i in range(numSegments):
            is_patch = CheckPatch(dctFeatures[:,:,i], fftFeatures[:,:,i], entropyFeatures[:,:,i])
    else:
        for i in range(numSegments):
            is_patch = CheckPatch(dctFeatures[:,:,i], fftFeatures[:,:,i])

    return numSegments-1