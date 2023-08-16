import modules.FeatureSegments as FS
import modules.utils as utils
import modules.NormalizingFlow as NF
import modules.RepaintModule as RM

import yaml
import ast
import numpy as np

def main():
    # Read Configurations
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    dataset_folder = config['dataset_folder']
    batch_size = config['batch_size']
    num_images = config['num_images']
    img_shape = ast.literal_eval(config['img_shape'])
    sam_model_path = config['sam_model_path']
    use_gpu = config['use_gpu']
    maximum_segments = config['maximum_segments']
    visualize = config['visualize']
    use_fft = config['use_fft']
    use_dct = config['use_dct']
    use_entropy = config['use_entropy']
    use_grayscale = config['use_grayscale']

    # Load SAM Model
    sam_model = utils.LoadSAMModel(modelPath=sam_model_path)
    sd2_model = RM.InitSD2()

    # Load the Dataset
    batch_data = utils.LoadApricotDataset(datasetPath=dataset_folder, batchsize=batch_size, numImages=num_images, shape=img_shape)

    all_imgs = []
    all_detected_patches = []
    all_inpainted_imgs = []

    for imgs, img_masks in batch_data:
        # Get Image Segment, DCT, FFT, Entropy, etc features
        img_info_list = FS.GetSegments(imgs=imgs, imgMasks=img_masks, samModel=sam_model, maximumSegments=maximum_segments, useFFT=use_fft,useDCT=use_dct, useEntropy=use_entropy, useGrayscale=use_grayscale)

        patch_indices = []
        patch_segments = []

        for i in range(len(imgs)):
            segment = img_info_list[i].SegmentMasks
            dct = img_info_list[i].Features.DCT
            fft = img_info_list[i].Features.FFT
            entropy = img_info_list[i].Features.Entropy

            # Get index of the patch among all the segments
            patch_idx = NF.GetSingleImagePatch(segment.shape[-1], dct, fft, entropy)

            patch_indices.append(patch_idx)
            detected_segment = np.uint8(255.0 * segment[:,:,patch_idx])
            patch_segments.append(detected_segment)

        # With the patch segment
        repainted_imgs = RM.getRepaintedImages(imgs, patch_segments, sd2_model)
        
        all_imgs = all_imgs + imgs
        all_detected_patches = all_detected_patches + patch_segments
        all_inpainted_imgs = all_inpainted_imgs + repainted_imgs

    return all_imgs, all_detected_patches, all_inpainted_imgs