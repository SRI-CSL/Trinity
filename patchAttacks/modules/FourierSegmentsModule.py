import cv2
import numpy as np

def GetFFTSingleSegment(maskedImage):
    # mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # masked_img = image * mask_3d
    gray_scale = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray_scale)
    magnitude_spectrum = np.abs(fft)
    return np.log(np.fft.fftshift(magnitude_spectrum) + 1)


# def GetFFTAllSegments(imgs, imgMasks, SAMModel, maximumSegments=20):
#     FFTSegment = namedtuple('FFTSegment', ['image', 'segments', 'ffts'])

#     fft_segment_list = []

#     for i in range(len(imgs)):
#         masks = SAMModel.generate(imgs[i])
#         all_masks = []
#         fft_list = []
#         masked_img_list = []

#         # Get Patch Index
#         for z, meta_info in enumerate(masks):
#             all_masks.append(meta_info['segmentation'])
        
#         iou_vals = utils.compute_vectorized_iou(imgMasks[i], all_masks)
#         patch_mask_idx = np.argmax(iou_vals)

#         # Non-patch
#         for j in range(min(len(masks), maximumSegments)):
#             if j != patch_mask_idx:
#                 current_mask = masks[j]['segmentation']
#                 masked_img, fft = GetFFTSingleSegment(imgs[i], current_mask)
#                 fft_list.append(fft)
#                 masked_img_list.append(masked_img)

#         # Patch
#         patch_mask = masks[patch_mask_idx]['segmentation']
#         masked_img, fft = GetFFTSingleSegment(imgs[i], patch_mask)
#         fft_list.append(fft)
#         masked_img_list.append(masked_img)

#         # Normalize the FFT
#         fft_list = np.stack(fft_list, axis=-1)
#         fft_list = 255.0 * (fft_list - np.min(fft_list)) / (np.max(fft_list) - np.min(fft_list))

#         fft_segment = FFTSegment(imgs[i], masked_img_list, fft_list)
#         fft_segment_list.append(fft_segment)

#     return fft_segment_list