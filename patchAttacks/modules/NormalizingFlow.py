def CheckPatch(dct=None, fft=None, entropy=None):
    return False

def GetSingleImagePatch(numSegments, dctFeatures, fftFeatures, entropyFeatures):
    if entropyFeatures.shape[0] != 0:
        for i in range(numSegments):
            is_patch = CheckPatch(dctFeatures[:,:,i], fftFeatures[:,:,i], entropyFeatures[:,:,i])
    else:
        for i in range(numSegments):
            is_patch = CheckPatch(dctFeatures[:,:,i], fftFeatures[:,:,i])

    return numSegments-1