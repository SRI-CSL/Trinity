import modules.FeatureSegments as FS
import modules.utils as utils

import yaml
import ast
import argparse
from tqdm import tqdm

def SaveImageFeatures(configPath):
    # Read Configurations
    with open(configPath) as f:
        config = yaml.safe_load(f)

    dataset_folder = config['dataset_folder']
    batch_size = config['batch_size']
    num_images = config['num_images']
    img_shape = ast.literal_eval(config['img_shape'])
    scale = config['scale']
    sam_model_path = config['sam_model_path']
    maximum_segments = config['maximum_segments']
    use_fft = config['use_fft']
    use_dct = config['use_dct']
    use_entropy = config['use_entropy']
    use_grayscale = config['use_grayscale']
    feature_save_folder = config['feature_save_folder']
    feature_shape = ast.literal_eval(config['feature_shape'])

    # Load SAM Model
    sam_model = utils.LoadSAMModel(modelPath=sam_model_path)

    batch_data = utils.LoadApricotDataset(datasetPath=dataset_folder, batchsize=batch_size, 
        numImages=num_images, shape=img_shape, scale=scale)

    for batch_num, (imgs, img_masks, img_names) in enumerate(batch_data):
        # Get Image Segment, DCT, FFT, Entropy, etc features
        img_info_list = FS.GetSegments(imgs=imgs, imgMasks=img_masks, samModel=sam_model, 
            maximumSegments=maximum_segments, useFFT=use_fft, 
            useDCT=use_dct, useEntropy=use_entropy, 
            useGrayscale=use_grayscale, featureShape=feature_shape, imageNames=img_names)

        for i in tqdm(range(len(imgs))):
            dct = img_info_list[i].Features.DCT
            fft = img_info_list[i].Features.FFT
            entropy = img_info_list[i].Features.Entropy
            image_name = img_info_list[i].ImageName
            utils.SaveFeatures(feature_save_folder, batch_num * batch_size + i, dct, fft, entropy, use_grayscale, image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line argument example")
    parser.add_argument("--config", help="config file")

    args = parser.parse_args()

    SaveImageFeatures(args.config)