from collections import namedtuple
import numpy as np

from . import utils
from . import CosineSegmentsModule
from . import FourierSegmentsModule
from . import EntropySegmentsModule


def GetSegments(imgs, imgMasks, samModel, maximumSegments=None, useFFT=True, useDCT=True, useEntropy=True):
    Features = namedtuple('Features', ['DCT', 'FFT', 'Entropy'])
    ImageInfo = namedtuple('ImageInfo', ['Image', 'SegmentMasks', 'Features'])

    img_info_list = []

    for i in range(len(imgs)):
        masks = samModel.generate(imgs[i])
        all_masks = []
        dct_list = []
        fft_list = []
        entropy_list = []
        masked_img_list = []

        # Get Patch Index
        for z, meta_info in enumerate(masks):
            all_masks.append(meta_info['segmentation'])
        
        iou_vals = utils.compute_vectorized_iou(imgMasks[i], all_masks)
        patch_mask_idx = np.argmax(iou_vals)

        # Non-patch
        non_patch_idx = 0

        total_segments = len(masks)
        if maximumSegments is not None:
            total_segments = maximumSegments
        
        for j in range(min(len(masks), total_segments)-1):
            if j < patch_mask_idx:
                non_patch_idx = j
            else:
                non_patch_idx = j+1

            current_mask = masks[non_patch_idx]['segmentation']
            mask_3d = np.repeat(current_mask[:, :, np.newaxis], 3, axis=2)
            masked_img = imgs[i] * mask_3d
            if useDCT:
                dct_segment = CosineSegmentsModule.GetDCTSingleSegment(masked_img)
                dct_list.append(dct_segment)
            if useFFT:
                fft_segment = FourierSegmentsModule.GetFFTSingleSegment(masked_img)
                fft_list.append(fft_segment)
            if useEntropy:
                entropy_segment = EntropySegmentsModule.GetEntropySingleSegment(masked_img)
                entropy_list.append(entropy_segment)
            masked_img_list.append(current_mask)

        # Patch
        patch_mask = masks[patch_mask_idx]['segmentation']
        mask_3d = np.repeat(patch_mask[:, :, np.newaxis], 3, axis=2)
        masked_img = imgs[i] * mask_3d

        if useDCT:
            dct_segment = CosineSegmentsModule.GetDCTSingleSegment(masked_img)
            dct_list.append(dct_segment)
        if useFFT:
            fft_segment = FourierSegmentsModule.GetFFTSingleSegment(masked_img)
            fft_list.append(fft_segment)
        if useEntropy:
            entropy_segment = EntropySegmentsModule.GetEntropySingleSegment(masked_img)
            entropy_list.append(entropy_segment)
        
        masked_img_list.append(patch_mask)

        masked_img_list = np.stack(masked_img_list, axis=-1)

        # Normalize the DCT
        if useDCT:
            dct_list = np.stack(dct_list, axis=-1)
            dct_list = 255.0 * (dct_list - np.min(dct_list)) / (np.max(dct_list) - np.min(dct_list))
        else:
            dct_list = np.array([])

        # Normalize the FFT
        if useFFT:
            fft_list = np.stack(fft_list, axis=-1)
            fft_list = 255.0 * (fft_list - np.min(fft_list)) / (np.max(fft_list) - np.min(fft_list))
        else:
            fft_list = np.array([])

        # Normalize the Entropy
        if useEntropy:
            entropy_list = np.stack(entropy_list, axis=-1)
            entropy_list = 255.0 * (entropy_list - np.min(entropy_list)) / (np.max(entropy_list) - np.min(entropy_list))
        else:
            entropy_list = np.array([])

        feature_segments = Features(dct_list, fft_list, entropy_list)
        img_info = ImageInfo(imgs[i], masked_img_list, feature_segments)
        img_info_list.append(img_info)

    return img_info_list