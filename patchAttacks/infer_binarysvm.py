import os
import glob
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import modules.utils as utils
from natsort import natsorted
import yaml
from tqdm import tqdm
import torchvision.models as models
from sklearn import svm
import pickle
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configfile = "configs/svm_config.yaml"
with open(configfile) as f:
    config = yaml.safe_load(f)

############ Feature Extractor ####################
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(vgg.features.children()))
feature_extractor = torch.nn.Sequential(*list(feature_extractor), torch.nn.Flatten(), *list(vgg.classifier.children())[:-6])
feature_extractor.to(device)
feature_extractor.eval()

data_dir = config['data_dir']

svm_model_files = os.listdir(config['svm_model'])

for i in range(len(svm_model_files)):
    svm_file = os.path.join(config['svm_model'], svm_model_files[i])
    # print(svm_file)
    with open(svm_file, 'rb') as file:
        clf = pickle.load(file)
    
    true_patches = 0
    detected_patches = 0

    for j in tqdm(range(138)):
        files = glob.glob(os.path.join(data_dir, "data_{}_*.png".format(j)))
        files = natsorted(files)

        images = utils.read_and_convert_to_tensor(files)

        # print("Extracting Features")
        images = images.to(device)
        with torch.no_grad():
            features = feature_extractor(images)

        features = features.to("cpu")

        # print("Features shape : ", features.shape)

        preds = clf.predict(features)
        # print(preds)
        detected_patches += np.sum(preds)
        true_patches += (preds[-1] == 1)

    print("For SVM Model : {}".format(svm_model_files[i]))
    print("True patches detected : ", true_patches * 100.0 / 138)
    print("Num Average Patches Detected : ", detected_patches/ 138)
    print("")