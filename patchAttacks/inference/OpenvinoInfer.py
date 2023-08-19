import os
import sys
from tqdm import tqdm
from PIL import Image

from anomalib.deploy import OpenVINOInferencer
from anomalib.data.utils import read_image
from anomalib.post_processing import Visualizer, VisualizationMode
from anomalib.data.task_type import TaskType


class OpenVinoInference():
    def __init__(self, modelFolder, modelName, metadataName):
        self.openvino_model_path = os.path.join(modelFolder, modelName)  
        self.metadata_path = os.path.join(modelFolder, metadataName)
        self.sanityCheck()
        self.visualizer = None
        self.inference_model = None

    
    def loadInferencer(self):
        self.inference_model = OpenVINOInferencer(
            path=self.openvino_model_path,  # Path to the OpenVINO IR model.
            metadata=self.metadata_path,  # Path to the metadata file.
            device="CPU",  # We would like to run it on an Intel CPU.
        )
    
    def sanityCheck(self):
        if os.path.isfile(self.openvino_model_path):
            print("OpenVino Model Found")
        else:
            print("Error : OpenVino Model Not Found at location : {}".format(self.openvino_model_path))
            sys.exit(1)
            
        if os.path.isfile(self.metadata_path):
            print("Metadata file found")
        else:
            print("Error : Metadata Model Not Found at location : {}".format(self.metadata_path))
            sys.exit(1)


    def openvinoInference(self, imagePath=None, image=None):
        if self.inference_model is None:
            self.loadInferencer()
        
        if imagePath is not None:
            image = read_image(imagePath)

        predictions = self.inference_model.predict(image=image)
        return predictions
    

    def visualizeImages(self, predictions):
        if self.visualizer is None:
            self.visualizer = Visualizer(mode=VisualizationMode.FULL, task=TaskType.CLASSIFICATION)
        
        for i in range(len(predictions)):
            output_image = self.visualizer.visualize_image(predictions[i])
            Image.fromarray(output_image)


    def infer(self, path, visualize=False, num_visualize=5, print_scores=False):
        images = []
        scores = []
        labels = []

        predictions = []

        if os.path.isfile(path):
            pred = self.openvinoInference(imagePath=path)
            predictions.append(pred)
            images.append(pred.image)
            scores.append(pred.pred_score)
            labels.append(pred.pred_label)

        elif os.path.isdir(path):
            file_paths = [entry.path for entry in os.scandir(path) if entry.is_file()]
            for i in tqdm(range(len(file_paths))):
                pred = self.openvinoInference(imagePath=file_paths[i])

                if i < num_visualize:
                    predictions.append(pred)
                
                images.append(pred.image)
                scores.append(pred.pred_score)
                labels.append(pred.pred_label)

        print("Total Images : {}".format(len(images)))
        print("Anomalies Detected (total): {}".format(sum(labels)))
        print("Anomalies Detected (in %): {:.2f}".format(sum(labels) * 100.0 / len(images)))
        
        if print_scores:
            for i in range(len(scores)):
                print("{} : {}".format(scores[i], labels[i]))

        if visualize:
            self.visualizeImages(predictions)
