import cv2
import numpy as np
import scipy


def GetEntropySingleSegment(maskedImage):
    # mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # masked_img = image * mask_3d
    gray_scale = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
    dct = scipy.entropy( scipy.entropy(gray_scale, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    magnitude_spectrum = np.log1p(dct)
    return magnitude_spectrum


# def GetEntropyAllSegments(imgs, imgMasks, SAMModel, maximumSegments=20):
#     DCTSegment = namedtuple('DCTSegment', ['image', 'segments', 'dcts'])

#     dct_segment_list = []

#     for i in range(len(imgs)):
#         masks = SAMModel.generate(imgs[i])
#         all_masks = []
#         dct_list = []
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
#                 masked_img, dct = GetEntropySingleSegment(imgs[i], current_mask)
#                 dct_list.append(dct)
#                 masked_img_list.append(masked_img)

#         # Patch
#         patch_mask = masks[patch_mask_idx]['segmentation']
#         masked_img, dct = GetEntropySingleSegment(imgs[i], patch_mask)
#         dct_list.append(dct)
#         masked_img_list.append(masked_img)

#         # Normalize the DCT
#         dct_list = np.stack(dct_list, axis=-1)
#         dct_list = 255.0 * (dct_list - np.min(dct_list)) / (np.max(dct_list) - np.min(dct_list))

#         dct_segment = DCTSegment(imgs[i], masked_img_list, dct_list)
#         dct_segment_list.append(dct_segment)

#     return dct_segment_list