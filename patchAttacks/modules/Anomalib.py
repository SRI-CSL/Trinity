import yaml
import os
from functools import partial, update_wrapper
from types import MethodType
from typing import Any
import ast
import sys

from anomalib.data import TaskType
from anomalib.data import InferenceDataset
from anomalib.data.folder import Folder
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer

from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode
from anomalib.post_processing import ThresholdMethod

from anomalib.models.fastflow.lightning_model import Fastflow
from anomalib.models.cflow.lightning_model import Cflow

class AnomalibModel():
    def __init__(self, configPath):
        self.config = self.readConfig(configPath)
        self.image_shape = ast.literal_eval(self.config["img_shape"])
        self.model = None
        self.inference_model = None

    def readConfig(self, configPath):
        with open(configPath) as f:
            config = yaml.safe_load(f)
        return config

    def setupDataModule(self):
        normal_dir = os.path.join(self.config["dataset_root"], self.config["normal_dir"])
        abnormal_dir = os.path.join(self.config["dataset_root"], self.config["abnormal_dir"])

        self.folder_datamodule = Folder(
            root=self.config["dataset_root"],
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            task=TaskType.CLASSIFICATION,
            image_size=self.image_shape[0],
            train_batch_size=self.config["train_batch_size"],
            eval_batch_size=self.config["eval_batch_size"],
            normalization=InputNormalizationMethod.NONE,
        )
        self.folder_datamodule.setup()
        self.folder_datamodule.prepare_data()
    
    def setupOptimizer(self):
        def configure_optimizers(lightning_module: LightningModule, optimizer: Optimizer) -> Any:  # pylint: disable=W0613,W0621
            """Override to customize the LightningModule.configure_optimizers` method."""
            return optimizer

        betas = ast.literal_eval(self.config["betas"])
        optimizer = Adam(params=self.model.parameters(), lr=self.config["lr"], betas=betas, weight_decay=float(self.config["weight_decay"]))
        fn = partial(configure_optimizers, optimizer=optimizer)
        update_wrapper(fn, configure_optimizers)  # necessary for `is_overridden`
        self.model.configure_optimizers = MethodType(fn, self.model)

    
    def setupCallbacks(self):
        callbacks = [
            MetricsConfigurationCallback(
                task=TaskType.CLASSIFICATION,
                image_metrics=["AUROC"],
            ),
            ModelCheckpoint(
                mode="max",
                monitor="image_AUROC",
            ),
            PostProcessingConfigurationCallback(
                threshold_method=ThresholdMethod.ADAPTIVE,
            ),
            ExportCallback(
                input_size=self.image_shape,
                dirpath=self.config["saved_model_dir"],
                filename=self.config["saved_model_name"],
                export_mode=ExportMode.OPENVINO,
            ),
        ]

        return callbacks
    

    def renameModel(self):
        base_folder = os.path.join(self.config["saved_model_dir"], "weights/openvino")
        openvino_file_name_old = os.path.join(base_folder, "model.bin")
        openvino_file_name_new = os.path.join(base_folder, self.config["saved_model_name"] + ".bin")

        metadata_file_name_old = os.path.join(base_folder, "metadata.json")
        metadata_file_name_new = os.path.join(base_folder, self.config["saved_model_name"] + ".json")
        os.rename(openvino_file_name_old, openvino_file_name_new)
        os.rename(metadata_file_name_old, metadata_file_name_new)



    def loadModel(self):
        if self.config["model_name"] == "cflow":
            self.model = Cflow(input_size=self.image_shape, backbone=self.config["model_backbone"], pre_trained=self.config["pre_trained_backbone"])
        elif self.config["model_name"] == "fastflow":
            self.model = Fastflow(input_size=self.image_shape, backbone=self.config["model_backbone"], pre_trained=self.config["pre_trained_backbone"], flow_steps=8)            


    def train(self):
        self.loadModel()
        self.setupOptimizer()
        callbacks = self.setupCallbacks()

        self.setupDataModule()

        self.trainer = Trainer(
            callbacks=callbacks,
            accelerator=self.config["accelerator"],
            devices=1,
            max_epochs=self.config["max_epochs"],
            logger=False,
        )

        self.trainer.fit(datamodule=self.folder_datamodule, model=self.model)
        # self.renameModel()


    def test(self):
        test_results = self.trainer.test(model=self.model, datamodule=self.folder_datamodule)

    def loadInferencer(self):
        openvino_model_path = os.path.join(self.config["saved_model_dir"], "weights/openvino", "model.bin")
        metadata_path = os.path.join(self.config["saved_model_dir"], "weights/openvino", "metadata.json")

        if os.path.isfile(openvino_model_path):
            print("OpenVino Model Found")
        else:
            print("Error : OpenVino Model Not Found at location : {}".format(openvino_model_path))
            sys.exit(1)
            
        if os.path.isfile(metadata_path):
            print("Metadata file found")
        else:
            print("Error : Metadata Model Not Found at location : {}".format(metadata_path))
            sys.exit(1)

        self.inference_model = OpenVINOInferencer(
            path=openvino_model_path,  # Path to the OpenVINO IR model.
            metadata=metadata_path,  # Path to the metadata file.
            device="CPU",  # We would like to run it on an Intel CPU.
        )

    def openvinoInference(self, imagePath=None, image=None):
        if self.inference_model is None:
            self.loadInferencer()
        
        if imagePath is not None:
            image = read_image(imagePath)

        predictions = self.inference_model.predict(image=image)
        return predictions
    

    def inference(self, folderPath):
        inference_dataset = InferenceDataset(path=folderPath, image_size=self.image_shape)
        inference_dataloader = DataLoader(dataset=inference_dataset)
        predictions = self.trainer.predict(model=self.model, dataloaders=inference_dataloader)

        num_true = sum(1 for p in predictions if p['pred_labels'].numpy()[0] == True)
        print("Anomalies Detected : {:.2f}".format(num_true * 100.0 / len(predictions)))