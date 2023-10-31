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

############ Training Set #############
train_dataset_path = "../../dataset/apricot_features/test"
print("Loading Training Set")
train_dataset = datasets.ImageFolder(
    train_dataset_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

train_features = []
train_labels = []

print("Loading Training Features")
for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
    images = images.to(device)
    with torch.no_grad():
        features = feature_extractor(images)
        train_features.append(features.to("cpu"))
        train_labels.append(labels)

train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)
train_labels_np = np.array(train_labels.clone())

print("Train Features shape : ", train_features.shape)
print("Train Labels shape : ", train_labels.shape)

######## Test Set ########
test_dataset_path = "../../dataset/apricot_features/dev"
print("Loading Test Set")
test_dataset = datasets.ImageFolder(
    test_dataset_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))


# test_subset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), 100, replace=False))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

test_features = []
test_labels = []

print("Loading Testing Features")
for i, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
    images = images.to(device)
    with torch.no_grad():
        features = feature_extractor(images)
        test_features.append(features.to("cpu"))
        test_labels.append(labels)

test_features = torch.cat(test_features, dim=0)
test_labels = torch.cat(test_labels, dim=0)
test_labels_np = np.array(test_labels.clone())

print("Test Features shape : ", test_features.shape)
print("Test Labels shape : ", test_labels.shape)


print("All features loaded, training SVMs")

C_s = [1.0, 1.5, 2.0, 2.5]
for C in C_s:
    clf = svm.SVC(C=C, kernel='rbf', gamma='scale', class_weight='balanced')
    clf.fit(train_features, train_labels)

    ## Training Prediction
    y_pred_train = clf.predict(train_features)
    outliers_accuracy = [y and y_pred for y,y_pred in zip(train_labels_np, y_pred_train)]
    inliers_accuracy = [not y and not y_pred for y,y_pred in zip(train_labels_np, y_pred_train)]
    print("For C : ", C)
    print("Training Accuracy for Outliers: " , np.sum(outliers_accuracy) * 100.0 / np.sum(train_labels_np))
    print("Training Accuracy for Inliers: " , np.sum(inliers_accuracy) * 100.0 / (len(train_labels_np) - np.sum(train_labels_np)))

    ## Test Prediction
    y_pred_test = clf.predict(test_features)
    outliers_accuracy = [y and y_pred for y,y_pred in zip(test_labels_np, y_pred_test)]
    inliers_accuracy = [not y and not y_pred for y,y_pred in zip(test_labels_np, y_pred_test)]
    print("Test Accuracy for Outliers: " , np.sum(outliers_accuracy) * 100.0 / np.sum(test_labels_np))
    print("Test Accuracy for Inliers: " , np.sum(inliers_accuracy) * 100.0 / (len(test_labels_np) - np.sum(test_labels_np)))
    print("--------------------------------------------------------------------------")
