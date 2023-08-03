import os
import torch
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import copy


def LoadApricotDataset(datasetPath = "/project/trinity/datasets/apricot/pub/apricot-mask/dev/data_mask_v2/", batchsize=32, shape=None):
    images_folder = datasetPath
    images_names = os.listdir(images_folder)

    imgs = []
    img_masks = []
    for i in range(len(images_names)):
        # idx = random.randint(0,len(gt_images_names)-1)
        img_info = torch.load(os.path.join(images_folder, images_names[i]))
        img = np.squeeze(img_info['Image'])
        img = np.uint8(img * 255.0)

        if shape is not None:
            img = cv2.resize(img, shape, interpolation = cv2.INTER_AREA)
        imgs.append(img)

        img_mask = np.squeeze(img_info['Mask'])
        img_mask = np.uint8(img_mask * 255.0)

        if shape is not None:
            img_mask = cv2.resize(img_mask, shape, interpolation = cv2.INTER_AREA)
        img_masks.append(img_mask)

        if len(imgs) == batchsize:
            yield imgs, img_masks
            imgs.clear()
            img_masks.clear()

    if len(imgs) > 0:
        yield imgs, img_masks


def compute_vectorized_iou(given_mask, masks):
    given_mask_expanded = np.expand_dims(given_mask, axis=0)
    masks_expanded = np.stack(masks)
    intersection = np.logical_and(masks_expanded > 0, given_mask_expanded > 0)
    union = np.logical_or(masks_expanded > 0, given_mask_expanded > 0)
    intersection_sum = np.sum(intersection, axis=(1, 2))
    union_sum = np.sum(union, axis=(1, 2))
    iou_values = intersection_sum / union_sum
    return iou_values


def LoadSAMModel(modelPath="/project/trinity/pretrained_models/sam_facebook/sam_vit_h_4b8939.pth", useGPU=True):
    sam = sam_model_registry["vit_h"](checkpoint=modelPath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if useGPU:
        sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def ConvertMaskToRectMask(maskList):
    rect_mask = []
    for i in range(len(maskList)):
        mask = copy.deepcopy(maskList[i])
        nonzero_indices = np.nonzero(mask)
        min_row = np.min(nonzero_indices[0])
        max_row = np.max(nonzero_indices[0])
        min_col = np.min(nonzero_indices[1])
        max_col = np.max(nonzero_indices[1])
        mask[min_row:max_row+1, min_col:max_col+1] = 255
        rect_mask.append(mask)
    return rect_mask


def ConvertToTensorList(npList):
    tensor_list = []
    for i in range(len(npList)):
        img = np.transpose(np.array(npList[i]), (2,0,1))
        img = torch.from_numpy(img.astype(np.float32))
        img /= 255.0
    return tensor_list


def ConvertToNumpyList(tensorList):
    np_list = []
    for i in range(len(tensorList)):
        img = (255.0 * np.transpose(tensorList[i].numpy(), (1,2,0))).astype(np.uint8)
        np_list.append(img)
    return np_list