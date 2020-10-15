from __future__ import print_function

import numpy as np

import torch
from torchvision import transforms

import argparse
import data_loader
import os
import yaml

from models import *

from compute_confidence import get_conf

# parse command line arguments for the config file
def parse_args(): 
    parser = argparse.ArgumentParser(description='Trinity ML Model Confidence Measurement')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the Trinity Confidence config file',
                        default='configs/Trinity_Confidence_config.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc, flush=True)

    return config



    
if __name__ == '__main__':
    # setting all gpus visible to this program
    os.environ['CUDA_VISIBLE_DEVICES']="0" #,1,2,3"
    
    # parse command line arguments for the config file
    config = parse_args()

    # if gpu is available then set device=gpu else set device=cpu
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format((config['trainer_params']['gpu'])))
        torch.cuda.manual_seed(0)
    else:
        device = torch.device('cpu')

    # set the path to pre-trained model and output
    pre_trained_net = config['model_params']['pretrained_model_path']
    if os.path.isdir(config['logging_params']['outf']) == False:
        os.mkdir(config['logging_params']['outf'])

    # create model
    model = classifier_models[config['model_params']['name']](config['exp_params']['num_classes'])

    # load pretrained model
    model.load_state_dict(torch.load(pre_trained_net, map_location = device))
    model.to(device)
    print('Loaded model: ' + config['model_params']['net_type'])
    
    # load dataset
    in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config['model_params']['transform_params']['mean'],config['model_params']['transform_params']['std'])])
    train_loader, test_loader = data_loader.getTargetDataSet(config['exp_params']['dataset'], config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'])
    print('Loaded dataset: ', config['exp_params']['dataset'])

    get_conf(config, device, model, in_transform, train_loader, test_loader)


