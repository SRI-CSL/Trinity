from modules import Anomalib

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line argument example")
    parser.add_argument("--config", help="model config file")

    args = parser.parse_args()
    model = Anomalib.AnomalibModel(modelconfigPath=args.config)
    model.train()
    model.test()