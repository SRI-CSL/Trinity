import os
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torchvision.datasets as datasets
from sklearn import svm
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading VGG pretrained model
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(vgg.features.children()))
# Take output from the last layer (4096 features)
feature_extractor = torch.nn.Sequential(*list(feature_extractor), torch.nn.Flatten(), *list(vgg.classifier.children())[:-6])
feature_extractor.to(device)
feature_extractor.eval()

# Loading Dataset (consisting of features)
#########################################
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset_path = "../../dataset/apricot_features/dev/dct_grayscale"
test_dataset_path = "../../dataset/apricot_features/dev/dct_grayscale_patch"

train_dataset = datasets.ImageFolder(
    train_dataset_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))

test_dataset = datasets.ImageFolder(
    test_dataset_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

train_features = []
test_features = []

print("Loading Training Features")
for i, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
    images = images.to(device)
    with torch.no_grad():
        features = feature_extractor(images)
        train_features.append(features.to("cpu"))

print("Loading Test Features")
for i, (images, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
    images = images.to(device)
    with torch.no_grad():
        features = feature_extractor(images)
        test_features.append(features.to("cpu"))

train_features = torch.cat(train_features, dim=0)
test_features = torch.cat(test_features, dim=0)

print("All features loaded, training SVMs")

nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for nu in nus:
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma='scale')
    clf.fit(train_features)

    ## Training Prediction
    y_pred_train = clf.predict(train_features)
    print("For nu : ", nu)
    print("Training Accuracy : " , np.sum(y_pred_train == 1) * 100.0 / len(y_pred_train))

    ## Test Prediction
    y_pred_test = clf.predict(test_features)
    print("Test Accuracy : " , np.sum(y_pred_test == -1) * 100.0 / len(y_pred_test))
    print("###################################")
