import os
import torch
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import copy
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image
import torch
from torchvision import transforms

def LoadImages(folderPath, batchsize=32, numImages=None, shape=None, scale=1.0):
    images_names = os.listdir(folderPath)
    imgs_list = []

    if numImages is None:
        numImages = len(images_names)
    for i in range(numImages):
        img = cv2.imread(os.path.join(folderPath, images_names[i]))
        h, w, _ = img.shape
        if shape is None:
            shape = (int(w * scale), int(h * scale))
        img = cv2.resize(img, shape, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgs_list.append(img)

        if len(imgs_list) == batchsize:
            yield imgs_list
            imgs_list.clear()

    if len(imgs_list) > 0:
        yield imgs_list


def LoadApricotDataset(datasetPath, batchsize=32, numImages=None, shape=None, scale=0.5):
    images_folder = datasetPath
    images_names = os.listdir(images_folder)

    imgs = []
    img_masks = []
    img_names = []

    if numImages is None or numImages > len(images_names):
        numImages = len(images_names)
    for i in range(numImages):
        # idx = random.randint(0,len(gt_images_names)-1)
        img_path = os.path.join(images_folder, images_names[i])
        img_name = img_path.split("/")[-1]
        img_name = img_name.split(".")[0]
        img_info = torch.load(img_path)
        img = np.squeeze(img_info['Image'])
        img = np.uint8(img * 255.0)

        img_names.append(img_name)
        if shape is None:
            shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, shape, interpolation = cv2.INTER_AREA)
        imgs.append(img)

        img_mask = np.squeeze(img_info['Mask'])
        img_mask = np.uint8(img_mask * 255.0)

        if shape is not None:
            img_mask = cv2.resize(img_mask, shape, interpolation = cv2.INTER_AREA)
        img_masks.append(img_mask)
  
        if len(imgs) == batchsize:
            yield imgs, img_masks, img_names
            imgs.clear()
            img_masks.clear()
            img_names.clear()

    if len(imgs) > 0:
        yield imgs, img_masks, img_names


def compute_vectorized_iou(given_mask, masks):
    given_mask_expanded = np.expand_dims(given_mask, axis=0)
    masks_expanded = np.stack(masks)
    intersection = np.logical_and(masks_expanded > 0, given_mask_expanded > 0)
    union = np.logical_or(masks_expanded > 0, given_mask_expanded > 0)
    intersection_sum = np.sum(intersection, axis=(1, 2))
    union_sum = np.sum(union, axis=(1, 2))
    iou_values = intersection_sum / union_sum
    return iou_values


def LoadSAMModel(modelPath, useGPU=True):
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


def VisualizeInpaintedOutputs(imgsList, patchList, inpaintedList):
    for i in range(len(inpaintedList)):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 20))
        axes[0].imshow(imgsList[i])
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        axes[1].imshow(patchList[i])
        axes[1].axis('off')
        axes[1].set_title('Detected Patch Mask')

        axes[2].imshow(inpaintedList[i])
        axes[2].axis('off')
        axes[2].set_title('Repainted Image')
    
    plt.tight_layout()
    plt.show()
    plt.close()

  
def VisualizeBboxes(imgsList, bboxList, labelsList, scoresList=None, labelsMapping=None, scoreThreshold = 0.3, numImagesToShow=5, numCols=3):
    images_displayed = 0
    if len(imgsList) < numImagesToShow:
        numImagesToShow = len(imgsList)
    
    while(images_displayed < numImagesToShow):
        num_images_current_row = min(numCols, numImagesToShow - images_displayed)
        fig, axes = plt.subplots(1, num_images_current_row, figsize=(3*num_images_current_row, 20))

        for j in range(num_images_current_row):
            if num_images_current_row == 1:
                axes.imshow(imgsList[images_displayed])
                axes.axis('off')
            else:  
                axes[j].imshow(imgsList[images_displayed])
                axes[j].axis('off')

            if scoresList is None:
                scores = [1.0 for i in range(bboxList[images_displayed].shape[0])]
            else:
                scores = scoresList[images_displayed]

            for k in range(bboxList[images_displayed].shape[0]):
                
                if scores[k] > scoreThreshold:
                    rectangle = patches.Rectangle((bboxList[images_displayed][k][0], bboxList[images_displayed][k][1]), bboxList[images_displayed][k][2], bboxList[images_displayed][k][3],linewidth=1, edgecolor='r', facecolor='none')
                    
                    if labelsMapping is None:
                        label = str(int(labelsList[images_displayed][k]))
                    else:
                        label = labelsMapping[int(labelsList[images_displayed][k])]

                    if num_images_current_row == 1:
                        axes.add_patch(rectangle)
                        axes.text(bboxList[images_displayed][k][0], bboxList[images_displayed][k][1], "{}:{:.3f}".format(label, scores[k]), fontsize=8, bbox=dict(facecolor='black', alpha=0.8, pad=1), color='white')
                    else:
                        axes[j].add_patch(rectangle)
                        axes[j].text(bboxList[images_displayed][k][0], bboxList[images_displayed][k][1], "{}:{:.3f}".format(label, scores[k]), fontsize=8, bbox=dict(facecolor='black', alpha=0.8, pad=1), color='white')
        
            images_displayed += 1
        plt.show()
        plt.close()


#TODO : Remove imgNumber

def SaveSingleFeature(folder, imgNumber, feature, featureName, useGrayscale, imageName):
    if useGrayscale:
        folder_clean = os.path.join(folder, featureName + "_grayscale")
        folder_patch = os.path.join(folder, featureName + "_grayscale_patch")
    else:
        folder_clean = os.path.join(folder, featureName + "_3d")
        folder_patch = os.path.join(folder, featureName + "_3d_patch")

    if not os.path.exists(folder_clean):
        os.makedirs(folder_clean)
    if not os.path.exists(folder_patch):
        os.makedirs(folder_patch)
    
    filename_patch = "{}.png".format(imageName)

    for i in range(feature.shape[-1]):
        if useGrayscale:
            save_feature = feature[:,:,i]
        else:
            save_feature = feature[:,:,:,i]
        if i == feature.shape[-1] - 1:
            cv2.imwrite(os.path.join(folder_patch, filename_patch), save_feature)
            continue
        filename = "{}_{}.png".format(imageName, i)
        cv2.imwrite(os.path.join(folder_clean, filename), save_feature)


def SaveFeatures(folder, imgNumber, dctList, fftList, entropyList, useGrayscale, imageName):
    if dctList.shape[0] != 0:
        SaveSingleFeature(folder, imgNumber, dctList, "dct", useGrayscale, imageName)
    if fftList.shape[0] != 0:
        SaveSingleFeature(folder, imgNumber, fftList, "fft", useGrayscale, imageName)
    if entropyList.shape[0] != 0:
        SaveSingleFeature(folder, imgNumber, entropyList, "entropy", useGrayscale, imageName)


def GetLabelsFromIndices(labelsFile):
    indices_to_labels = []
    with open(labelsFile, "r") as file:
        for line in file:
            indices_to_labels.append(line.strip())
    return indices_to_labels


def GetApricotAnnotations(apricotAnnotationPath):
    annotations_file = apricotAnnotationPath
    with open(annotations_file, 'r') as file:
        data = json.load(file)

    annotations_coco_91 = {}

    for i in range(len(data['categories'])):
        annotations_coco_91[data['categories'][i]['id']] = data['categories'][i]['name']

    return annotations_coco_91


def ConvertToTensorList(npList):
    tensor_list = []
    for i in range(len(npList)):
        img = np.transpose(np.array(npList[i]), (2,0,1))
        img = torch.from_numpy(img.astype(np.float32))
        img /= 255.0
        tensor_list.append(img)
    return tensor_list


def ConvertToNumpyList(tensorList):
    np_list = []
    for i in range(len(tensorList)):
        img = (255.0 * np.transpose(tensorList[i].numpy(), (1,2,0))).astype(np.uint8)
        np_list.append(img)
    return np_list


def generateSegments(samModel, image):
    masks = samModel.generate(image)
    all_masks = []
    for z, meta_info in enumerate(masks):
        all_masks.append(meta_info['segmentation'])
    
    return all_masks


def read_and_convert_to_tensor(file_names):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    images = []
    
    for i in range(len(file_names)):
        file_path = file_names[i]
        # print(file_path)
        image = Image.open(file_path).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)

    return torch.stack(images)