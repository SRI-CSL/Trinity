import inference.OpenvinoInfer as Openvino

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line argument example")
    parser.add_argument("--modelFolder", help="Folder that contains the model and metadata file. It assumes that model file is named model.bin and metadata as metadata.json. Else see --modelName and --metadataName")
    parser.add_argument("--modelName", default="model.bin", help="Name of model file - e.g. model.bin, fastflow.bin, cflow.bin")
    parser.add_argument("--metadataName", default="metadata.json", help="Name of metadata file")
    parser.add_argument("--visualize", action="store_true", help="Visualize the Inferred Images")
    parser.add_argument("--numVisualize", type=int, default=5, help="If visualize flag is passed, specify the number of images to visualize")
    parser.add_argument("--inferPath", help="Give the path of image to infer OR the path of entire folder")
    parser.add_argument("--printScores", action="store_true", help="Print the predicted scores")

    args = parser.parse_args()

    openvino_infer = Openvino.OpenVinoInference(modelFolder=args.modelFolder, modelName=args.modelName, metadataName=args.metadataName)
    openvino_infer.infer(path=args.inferPath, visualize=args.visualize, num_visualize=args.numVisualize, print_scores=args.printScores)