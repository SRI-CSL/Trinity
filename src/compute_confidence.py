from __future__ import print_function

import numpy as np

import torch
from torchvision import transforms
from torch.autograd import Variable

import data_loader
import os

import calculate_log as callog
import lib_generation
from models import *

import getFeatures
import getStats


def get_conf(config, device, model, in_transform, train_loader, test_loader):
    
    # get layer list
    layer_list = config['exp_params']['feature_layers'] 


    # get feature list
    model.eval()
    input_dim = config['exp_params']['input_dim'] # 2 x 3 x 32 x 32

    if config['exp_params']['dataset'] != 'toy_data' :
        temp_x = torch.rand(2, input_dim[0], input_dim[1], input_dim[2]).to(device)
    else:
        temp_x =  torch.rand(2,2).to(device)
    _, temp_list = model.feature_list(temp_x, layer_list)
    
    # ASK : Why this round about way to get feature_list size? Using temp_x etc.
    # RESPONSE: number of layers
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    # m_list is the list of noise
    m_list = config['exp_params']['noise_params']['m_list']

    # calculate confidence components to be sent to regressor 
    regressor_features = config['exp_params']['regressor_features']
    class_mean, class_precision, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision = \
                get_inputs_for_computing_regressor_feature(regressor_features, model, config, num_output, feature_list, layer_list, train_loader, device)
            



    print("For in-distribution: {}".format(config['exp_params']['dataset']))
    init_reg_in = True
    for regressor_feature in regressor_features:
        # num_output is the number of layers
        for i in range(num_output):
            in_dist_input = get_features_for_regressor(regressor_feature, model, config, test_loader, config['exp_params']['dataset'], i, True, device,
                                                class_mean, class_precision, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision)

            print("in_dist_input shape: ", in_dist_input.shape)
            in_dist_input = np.asarray(in_dist_input, dtype=np.float32)

            # ASK: what does score of regression mean?
            print("Mean score at layer {} for regression type {}: {}".format(i,regressor_feature,np.mean(in_dist_input)))

            if init_reg_in:
                regressor_in_dist_input = in_dist_input.reshape((in_dist_input.shape[0], -1))
                init_reg_in = False
            else:
                regressor_in_dist_input = np.concatenate((regressor_in_dist_input, in_dist_input.reshape((in_dist_input.shape[0], -1))), axis=1)

    print("Out-distributions to test agains: ", config['model_params']['out_dist_list'])        
    for out_dist in config['model_params']['out_dist_list']:
        print('Out-distribution: ' + out_dist)
        if out_dist == 'subset_cifar100':
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'], idx=config['model_params']['out_idx'], num_oods=config['model_params']['num_ood_samples'])
        else:
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'])
        
        init_reg_in = True
        for regressor_feature in regressor_features:
            # num_output is the number of layers
            for i in range(num_output):
                out_dist_input = get_features_for_regressor(regressor_feature, model, config, out_test_loader, out_dist,  i, False, device,
                                                    class_mean, class_precision, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision)

                print("out_dist_input shape- ", out_dist_input.shape)
                out_dist_input = np.asarray(out_dist_input, dtype=np.float32)
                print("Mean score at layer {} for regression type {}: {}".format(i,regressor_feature,np.mean(out_dist_input)))
                if init_reg_in:
                    regressor_out_dist_input = out_dist_input.reshape((out_dist_input.shape[0], -1))
                    init_reg_in = False
                else:
                    regressor_out_dist_input = np.concatenate((regressor_out_dist_input, out_dist_input.reshape((out_dist_input.shape[0], -1))), axis=1)
            
        regressor_in_dist_input = np.asarray(regressor_in_dist_input, dtype=np.float32)
        regressor_out_dist_input = np.asarray(regressor_out_dist_input, dtype=np.float32)
        ood_output, Mahalanobis_labels = lib_generation.merge_and_generate_labels(regressor_out_dist_input, regressor_in_dist_input)
        file_name = os.path.join(config['logging_params']['outf'], 'Mahalanobis_%s_%s_%s.npy' % (str(m_list[0]), config['exp_params']['dataset'] , out_dist))
        ood_output = np.concatenate((ood_output, Mahalanobis_labels), axis=1)
        np.save(file_name, ood_output)
    return ood_output





def get_inputs_for_computing_regressor_feature(regressor_features, model, config, num_output, feature_list, layer_list, train_loader, device):
    
    num_classes = config['exp_params']['num_classes']

    class_mean, class_precision, tied_precision, knn_mean, knn_precision = None, None, None, None, None
    knn_search_list = [None]*num_output
    pca_list = []
    
    for regressor_feature in regressor_features:
        
        list_features, _ = getFeatures.get_all_features(model, num_classes, feature_list, layer_list, train_loader, device)
        feature_list = []
        # assume: ordering respected
        for layer in range(0,len(layer_list)):
            # RESPONSE: list_features is num_layers X num_classes X tensor ( num_data_points x feature_length_for_each_layer)
            feature_list.append(list_features[layer][0].shape[1])
            print("In feature_list creation:", list_features[layer][0].shape[1])


        if regressor_feature == 'mahalanobis_tied_cov':
            print("get mean and precision for {}".format(regressor_feature))
            class_mean, tied_precision = getStats.calc_class_mean_tied_precision(list_features, model, num_classes, feature_list, train_loader, device)
        
        # we need class_mean and class_precision for introducing noise in the input features acc to the Mahalanobis paper
        if regressor_feature == 'mahalanobis_class_cov' or regressor_feature == 'pca' and class_precision == None :
            print("get mean and precision for {}".format(regressor_feature))
            class_mean, class_precision = getStats.calc_class_mean_class_precision(list_features, model, num_classes, feature_list, train_loader, device)
        
        if regressor_feature == 'pca':
            print("get pca_list for {}".format(regressor_feature))
            pca_list = getStats.get_pca(list_features, model, num_classes, feature_list, train_loader, device)
        
        if regressor_feature == 'knn_mahalanobis_class_cov' or regressor_feature == 'knn_mahalanobis_tied_cov':     
            print("get knn_serch_list, mean and precision for {}".format(regressor_feature))

            knn_args = config['exp_params']['knn_args']
            if regressor_feature == 'knn_mahalanobis_class_cov':
                knn_cov_type = 'class_cov'
                # for adding noise
                if class_precision == None:
                    class_mean, class_precision = getStats.calc_class_mean_class_precision(list_features, model, num_classes, feature_list, train_loader, device)

            else:
                knn_cov_type = 'tied_cov'
                # for adding noise
                if tied_precision == None:
                    class_mean, tied_precision = getStats.calc_class_mean_tied_precision(list_features, model, num_classes, feature_list, train_loader, device)

            # set the KNNSearch class params according to the knn search library type
            if knn_args['knn_type'] =='datasketch':
                knn_type_args = {'algorithm':'datasketch', 'num_perm':128,'normalize':False,'create':False,'file_path':'datasketch.pkl'}
            elif knn_args['knn_type'] =='annoy':
                knn_type_args = {'algorithm':'annoy','num_trees':128,'normalize':False,'metric':'euclidean','create':True,'file_path':'annoy.ann'}
            elif knn_args['knn_type'] =='exact':
                knn_type_args = {'algorithm':'exact','normalize':False}
            elif knn_args['knn_type'] =='random':
                knn_type_args = {'algorithm': 'random','normalize':False}
            elif knn_args['knn_type'] =='falconn':
                knn_type_args = {'algorithm': 'falconn','number_bits':17,'nb_tables':200,'normalize':True}
            elif knn_args['knn_type'] =='descent':
                knn_type_args = {'algorithm': 'descent','metric':'euclidean', 'normalize':False}
            else:
                raise Exception('Wrong KNN algorithm input')
            
            # feature_list for knn will have different dimension than the original feature_list
            if config['exp_params']['dataset'] != 'toy_data' :
                temp_x = torch.rand(2,3,32,32).to(device)
            else:
                temp_x =  torch.rand(2,2).to(device)            
            #temp_x = Variable(temp_x)
            temp_list = model.feature_list(temp_x, layer_list)[1]
            knn_feature_list = np.zeros(num_output)
            count = 0
            for out in temp_list:
                if config['exp_params']['knn_args']['keep_original']:
                    knn_feature_list[count] = out.size(1)
                if config['exp_params']['knn_args']['keep_knn_mean']:
                    knn_feature_list[count]+=out.size(1)
                if config['exp_params']['knn_args']['keep_knn_std']:
                    knn_feature_list[count]+=out.size(1)
                count += 1

            print("Calling calc_knn_mean_precision")
            knn_search_list, knn_mean, knn_precision = getStats.calc_knn_mean_precision(list_features = list_features, 
                                                                                    model = model, 
                                                                                    num_classes = num_classes,
                                                                                    feature_list = knn_feature_list, 
                                                                                    train_loader = train_loader,
                                                                                    device = device, 
                                                                                    cov_type = knn_cov_type, 
                                                                                    knn_type_args = knn_type_args, 
                                                                                    knn_args = knn_args)
            print("Done calc_knn_mean_precision")
    return class_mean, class_precision, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision


def get_features_for_regressor(regressor_feature, model, config, test_loader, dataset, i, out_flag, device,
                               class_mean, class_precision, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision):

    if regressor_feature == 'mahalanobis_class_cov':

        print("Getting scores using Mahalanobis class covoriance")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, class_precision,
                                                    class_mean, class_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=False)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    elif regressor_feature == 'mahalanobis_tied_cov':

        print("Getting scores using Mahalanobis tied covoriance [Mahalanobis Paper]")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, tied_precision,
                                                    class_mean, tied_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=False)
            cur_score = np.array(cur_score)            
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)
    elif regressor_feature == 'pca':

        print("Getting scores using PCA")        
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_pca_score(model, config, test_loader, out_flag,
                                                    class_mean, class_precision, i, magnitude, pca_list, device)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    elif regressor_feature == 'knn_mahalanobis_class_cov':

        print("Getting scores using Mahalanobis class covoriance on K-NNs")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, class_precision,
                                                    knn_mean, knn_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=True)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)
        
    elif regressor_feature == 'knn_mahalanobis_tied_cov':

        print("Getting scores using Mahalanobis tied covoriance on K-NNs")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, tied_precision,
                                                    knn_mean, knn_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=True)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    elif regressor_feature == 'ODIN':
        scores = []
        for params in config['exp_params']['odin_args']['settings']:
            cur_score = lib_generation.get_posterior(model, config['model_params']['net_type'], test_loader, params[1], params[0], config['logging_params']['outf'], out_flag, device)
            scores.append(cur_score.reshape(-1,1))

        return np.hstack(scores)

    elif regressor_feature == 'LID':
        # dumping code
        os.system("python ADV_Samples.py --dataset {} --net_type {} --adv_type {} --gpu {} --outf {} --model {} --ood_idx {} --num_oods {}".format(dataset,
        config['model_params']['net_type'], config['exp_params']['lid_args']['adv_type'], config['trainer_params']['gpu'], config['exp_params']['lid_args']['outf'], 
        config['exp_params']['dataset'], config['model_params']['out_idx'], config['model_params']['num_oods']))
        # scoring code
        base_path = config['exp_params']['lid_args']['outf'] + config['model_params']['net_type'] + '_' + dataset + '/'
        test_clean_data = torch.load(base_path + 'clean_data_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
        test_adv_data = torch.load(base_path  + 'adv_data_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
        test_noisy_data = torch.load(base_path  + 'noisy_data_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
        test_label = torch.load(base_path + 'label_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
        LID, LID_adv, LID_noisy = lib_generation.get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, i+1)
        LID_scores = np.hstack([np.vstack(s) for s in LID])
        print("LID_scores_shape:",LID_scores.shape)
        return LID_scores

    else:
        raise Exception("Wrong type of regressor feature")







