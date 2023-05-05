from scipy.io import loadmat
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def retbox(bbox, format='xyxy'):
    """A utility function to return box coords asvisualizing boxes."""
    if format == 'xyxy':
        xmin, ymin, xmax, ymax = bbox
    elif format == 'xywh':
        xmin, ymin, w, h = bbox
        xmax = xmin + w - 1
        ymax = ymin + h - 1

    box = np.array([[xmin, xmax, xmax, xmin, xmin],
                    [ymin, ymin, ymax, ymax, ymin]])
    return box.T

class Loader(object):
    def __init__(self):
        pass

    def convert_and_maybe_resize(self, im, resize):
        scale = 1.0
        im = np.asarray(im)
        if resize:
            h, w, _ = im.shape
            scale = min(1000 // max(h, w), 600 // min(h, w))
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(im), scale


import data.transforms as T


def get_transform(istrain=False):
    transforms = []
    transforms.append(T.ToTensor())
    if istrain:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

imgtransform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def showAnns(ax, anns):
    """
        ax: matplotlib axes
        anns: list of Nx2 arrays of polygons coordinates  (x,y)
    """

    polygons = []
    color = []
    for poly in anns:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        polygons.append(Polygon(poly))
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


class OOCD(Loader):

    def __init__(self, root='./datasets/' , included=[]):
        super().__init__()
        self.root = Path(root)
        gt_path = self.root / "new_outofcontext_groundtruth.mat"
        data = loadmat(gt_path, simplify_cells=True)  # necessary if there are arrays inside mat files
        self.data = data['Doutofcontext']
        self.image_dir = Path("/data/diva-1/manoj/open_world_datasets/Sun_images/out_of_context")
        # 111 classes
        # categoriespath = "/data/diva-1/manoj/open_world_datasets/dataset/sun09_objectCategories.mat"
        # cats = loadmat(categoriespath, simplify_cells=True)
        # self.CLASSES = cats['names'].tolist()
        # 576 classes
        sungtpath = "/data/diva-1/manoj/open_world_datasets/dataset/sun09_groundTruth.mat"
        sungt = loadmat(sungtpath, simplify_cells=True)
        self.CLASSES = sungt['categories']['names'].tolist()

    def __len__(self):
        return len(self.data)

    def convert(self, ann):

        objects = ann['annotation']['object']
        fname = ann['annotation']['filename']
        polys = []
        oocs = {}

        boxes = []
        classes = []
        area = []
        iscrowd = []
        isoocs = []

        H, W = ann['annotation']['imagesize']['nrows'], ann['annotation']['imagesize']['ncols']

        for idx, obj in enumerate(objects):
            x, y = obj['polygon']['x'], obj['polygon']['y']
            xy = np.vstack((x, y)).T  # N x 2 needed
            isooc = obj['outofcontext']
            name = obj['name']
            try:
                cat = self.CLASSES.index(name)
            except:
                cat = 999
            # cat = obj['id'] #just a increasing id number
            xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
            bbox = [xmin, ymin, xmax, ymax]
            difficult = 0
            boxes.append(bbox)
            classes.append(cat)
            iscrowd.append(difficult)
            isoocs.append(isooc)
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)
        isoocs = torch.as_tensor(isoocs)

        image_id = fname.replace("im","").rstrip(".jpg") #'im201.jpg'
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes.long()
        target["image_id"] = image_id
        target["size"] = torch.as_tensor([int(H), int(W)])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["isooc"] = isoocs

        return  target

    def show(self, index):
        ann = self.data[index]
        fname = ann['annotation']['filename']
        objects = ann['annotation']['object']
        polys = []
        for idx, obj in enumerate(objects):
            x, y = obj['polygon']['x'], obj['polygon']['y']
            xy = np.vstack((x, y)).T  # N x 2 needed
            isooc = obj['outofcontext']
            name = obj['name']
            polys.append(xy)
            if isooc:
                xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
                bbox = np.array([[xmin, ymin],
                                 [xmin, ymax],
                                 [xmax, ymax],
                                 [xmax, ymin],
                                 ])
                polys.append(bbox)


        # using the variable ax for single a Axes
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        pil = Image.open(self.image_dir / fname)
        ax[0].axis('off')
        ax[0].imshow(pil)
        ax[1].imshow(pil)
        ax[1].set_autoscale_on(False)
        ax[1].axis('off')
        showAnns(ax[1], polys)


    def __getitem__(self, index):
        ann = self.data[index]
        fname = ann['annotation']['filename']
        img = pil_loader(self.image_dir / fname)
        #add image size here
        ann['annotation']['imagesize'] = { 'nrows': img.height , 'ncols': img.width}
        return imgtransform(img), self.convert(ann)

    def item(self, index):
        ann = self.data[index]
        fname = ann['annotation']['filename']
        img = pil_loader(self.image_dir / fname)
        #add image size here
        ann['annotation']['imagesize'] = { 'nrows': img.height , 'ncols': img.width}
        return self.convert(ann)


if __name__ == '__main__':

    loader = OOCD(root="datasets/supportContextPublic/data/")
    L = len(loader)
    import random
    index = random.randint(0,L)
    print(loader[index])
    loader.show(index)
    plt.show()

    s , ooc = 0
    for i,data in enumerate(loader):
        _ , ann = data
        s +=len(ann['boxes'])
        ooc += sum(ann['isooc'])
    print ( "total images", len(loader) , "total instances" , s ,"total ooc instances" , ooc )