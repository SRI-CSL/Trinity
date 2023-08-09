from pytorchyolo import detect, models
from modules import utils

class YOLOModel:
    def __init__(self, 
                 cfgPath = "/project/trinity/pretrained_models/yolo/yolov3-tiny/yolov3-tiny.cfg", 
                 modelPath="/project/trinity/pretrained_models/yolo/yolov3-tiny/yolov3-tiny.weights",
                 labelsFile="/project/trinity/datasets/COCO/coco_2014.names"):
        self.cfg_path = cfgPath
        self.model_path = modelPath
        self.labels_file = labelsFile
        self.bbox_list = []
        self.labels_list = []
        self.scores_list = []
        
        self.loadYOLOModel()
        self.getLabelsFromIndices()
    
    def getLabelsFromIndices(self):
        self.indices_to_labels = utils.GetLabelsFromIndices(self.labels_file)
        
    def loadYOLOModel(self):
        self.model = models.load_model(self.cfg_path, self.model_path)

    def inference(self, imgsList):
        self.bbox_list.clear()
        self.labels_list.clear()
        self.scores_list.clear()
        self.imgs_list = imgsList

        for i in range(len(self.imgs_list)):
            bboxes = detect.detect_image(self.model, imgsList[i])
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

            labels = bboxes[:, 5]
            scores = bboxes[:, 4]
            
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
