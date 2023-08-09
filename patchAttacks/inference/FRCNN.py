import copy
import numpy as np
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from modules import utils


class FRCNNModel:
    def __init__(self, labelsFile="/project/trinity/datasets/COCO/coco_original.names"):
        self.labels_file = labelsFile
        self.bbox_list = []
        self.labels_list = []
        self.scores_list = []
        self.loadFRCNNModel()
        self.getLabelsFromIndices()
    

    def getLabelsFromIndices(self):
        self.indices_to_labels = utils.GetLabelsFromIndices(self.labels_file)

    
    def loadFRCNNModel(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()


    def inference(self, imgsList):
        self.bbox_list.clear()
        self.labels_list.clear()
        self.scores_list.clear()
        self.imgs_list = imgsList

        if isinstance(imgsList[0], np.ndarray):
            imgsList = utils.ConvertToTensorList(imgsList)

        predictions = self.model(imgsList)

        for i in range(len(imgsList)):
            bboxes = copy.deepcopy(predictions[i]['boxes'].detach().numpy())
            bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
            bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])

            labels = predictions[i]['labels'].detach().numpy()
            labels = labels - 1
            scores = predictions[i]['scores'].detach().numpy()

            self.bbox_list.append(bboxes)
            self.labels_list.append(labels)
            self.scores_list.append(scores)

        return self.bbox_list, self.labels_list, self.scores_list
    

    def visualize(self, showLabels=True, scoreThreshold = 0.3, numImagesToShow=None, numCols=3):
        if numImagesToShow is None:
            numImagesToShow = len(self.imgs_list)

        idx_to_labels = None
        if showLabels:
            idx_to_labels = self.indices_to_labels
        utils.VisualizeBboxes(self.imgs_list, self.bbox_list, self.labels_list, self.scores_list, labelsMapping=idx_to_labels, scoreThreshold = scoreThreshold, numImagesToShow=numImagesToShow, numCols=numCols)
