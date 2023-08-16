## Running Instructions

The pipeline is divided into 3 parts - 

- Step 1 : Generating Features (DCT, FFT, Entropy) and saving those as images
- Step 2 : Run the Normalizing Flow (training and inference)
- Step 3 : Run the Repaint method using the trained normalizing flow model

### Step 1 - Generating Features

```
python save_features.py --config configs/config.yaml
```
The config file contains the dataset folder, batch size, image size and other information.

### Step 2 - Training the Normalizing Flow

```
python train.py --config configs/model/fastflow.yaml
```
train.py requires a model configuration file (since there are multiple algorithms in normalizing flow). The config file also contains the path to the saved features (generated in the previous step).

#### Step 2.1 Inferencing

Once we have the trained model, we can use it for inference. 

```
python openvino_inference.py --modelFolder /project/trinity/pretrained_models/norm_flow/fastflow/weights/openvino/ --inferPath /project/trinity/datasets/apricot_features/dev/dct_gs_patch/ --printScores
```

openvino_inference.py requires the folder containing the model (and metadata file), image path or folder of images that are to be inferred.
