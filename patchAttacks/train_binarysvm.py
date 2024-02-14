import os
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torchvision.datasets as datasets
from sklearn import svm
import numpy as np
from tqdm import tqdm
import yaml
import pickle
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Config
configfile = "configs/svm_config.yaml"
with open(configfile) as f:
    config = yaml.safe_load(f)


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


########### Training Set #############
train_dataset_path = config['train_dataset']
print("Loading Training Set")

# 1 -> Anomalies, 0 -> Natural Segments

train_dataset = datasets.ImageFolder(
    train_dataset_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

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
exit(0)

####### Test Set ########
test_dataset_path = config['test_dataset']
print("Loading Test Set")
test_dataset = datasets.ImageFolder(
    test_dataset_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]))


# test_subset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), 100, replace=False))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

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
# print(np.sum(test_labels_np == 1))
print("Test Labels shape : ", test_labels.shape)


print("All features loaded, training SVMs")

if not os.path.exists(config["save_model_folder"]):
    os.makedirs(config["save_model_folder"])

C_s = [1.0, 1.5, 2.0, 2.5]
for C in C_s:
    start = time.time()
    clf = svm.SVC(C=C, kernel='rbf', gamma='scale', class_weight='balanced')
    clf.fit(train_features, train_labels)

    ## Training Prediction
    print("Training ....")
    y_pred_train = clf.predict(train_features)
    anomalies_accuracy = [y and y_pred for y,y_pred in zip(train_labels_np, y_pred_train)]
    natural_segments_accuracy = [not y and not y_pred for y,y_pred in zip(train_labels_np, y_pred_train)]
    print("For C : ", C)
    print("Training Accuracy for Anomalies: " , np.sum(anomalies_accuracy) * 100.0 / np.sum(train_labels_np))
    print("Training Accuracy for Natural Segments: " , np.sum(natural_segments_accuracy) * 100.0 / (len(train_labels_np) - np.sum(train_labels_np)))

    ## Test Prediction
    y_pred_test = clf.predict(test_features)
    anomalies_accuracy = [y and y_pred for y,y_pred in zip(test_labels_np, y_pred_test)]
    natural_segments_accuracy = [not y and not y_pred for y,y_pred in zip(test_labels_np, y_pred_test)]
    print("Test Accuracy for Anomalies: " , np.sum(anomalies_accuracy) * 100.0 / np.sum(test_labels_np))
    print("Test Accuracy for Natural Segments: " , np.sum(natural_segments_accuracy) * 100.0 / (len(test_labels_np) - np.sum(test_labels_np)))
    print("--------------------------------------------------------------------------")

    # SAving Models
    print("Saving Models ....")
    model_name = os.path.join(config["save_model_folder"], "svm_{}.pkl".format(C))

    with open(model_name,'wb') as f:
        pickle.dump(clf,f)

    print("Model Saved at {}".format(model_name))
    end = time.time()
    print("Time taken : {}".format(end - start))
    print("")