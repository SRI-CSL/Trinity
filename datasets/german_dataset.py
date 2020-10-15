# The German Traffic Sign Recognition Benchmark (GTSRB) Dataset
import os
import numpy as np
import PIL
from six.moves import range
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from skimage import io, transform

class GermanTrafficData(Dataset):

    def __init__(self, root, img_size, train=True):
        # Creating the list of images and corresponding labels 
        images = [] # images
        labels = [] # corresponding labels
        self.transform = transform
        self.train = train
        # training data with all the 43 classes, each folder contains training images of one class
        if (self.train):
            start_index_folder = 0
            end_index_folder = 43
            # loop over classes
            for c in range(start_index_folder, end_index_folder):
                prefix = root + '/' + format(c, '05d') + '/' # subdirectory for class
                gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
                gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
                next(gtReader) # skip header
                # loop over all images in current annotations file
                for row in gtReader:
                    images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                    labels.append(int(row[7])) # the 8th column is the label
                gtFile.close()
        # test data with all images and csv in one folder, whose path is specified in root
        else:
            prefix = root + '/'
            gtFile = open(prefix + 'GT-final_test.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                labels.append(int(row[7])) # the 8th column is the label
            gtFile.close()

        # preprocess image size to self.img_size X self.img_size
        self.img_size = img_size
        # assign all images to the class object after preprocessing them in the format required by the forward function  
        self.images = torch.from_numpy(self.all_resize_images(np.array(images)))
        self.labels = torch.from_numpy(np.array(labels))

    def __len__(self):
        return len(self.images)
        
    def observation_shape(self):
        return [3, self.img_size, self.img_size]
    
    def __getitem__(self, idx):

        sample = self.images[idx]
        
        return sample, self.labels[idx]
    
    # Resizing all images to 3 X self.img_size X self.img_size and normalizing them in [0, 1]
    def all_resize_images(self, orig_images):
        resized_images = np.zeros((orig_images.shape[0], 3, self.img_size, self.img_size), np.float32)
        for i,img in enumerate(orig_images):
            image = PIL.Image.fromarray(img)
            image = image.resize((self.img_size, self.img_size), PIL.Image.ANTIALIAS)
            resized_images[i,:,:,:] = np.transpose(np.array(image), (2, 0, 1))
        
        resized_images/=255.
        return resized_images

# unit test for this dataset class- loads an image with random index and displays it
def test():
    dataset = GermanTrafficData(root='datasets/GTSRB-Test/Final_Test/Images', img_size=64, train = False)
    len = dataset.__len__()
    index = random.randint(0, len)
    img, label = dataset[index]
    print(label)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    test()
