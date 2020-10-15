from __future__ import print_function

import os
import numpy as np
import torch
import pickle

from datasketch import MinHashLSHForest
from datasketch import MinHash

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.decomposition import PCA
import sklearn.covariance

import argparse
import data_loader
import calculate_log as callog
import models
import lib_generation

from torchvision import transforms
from torch.autograd import Variable

from annoy import AnnoyIndex

from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

from pynndescent import NNDescent

import getFeatures
import knnSearch 
from knnSearch import KNNSearch
import random
random.seed(20)


def get_precision(features):
    '''
    Compute precision
    '''

    # ASK: MinCovDet  vs EmpiricalCovariance

    # The matrix inverse of the covariance matrix, often called the precision matrix, 
    # is proportional to the partial correlation matrix. It gives the partial independence 
    # relationship. In other words, if two features are independent conditionally on the 
    # others, the corresponding coefficient in the precision matrix will be zero. This is 
    # why it makes sense to estimate a sparse precision matrix: the estimation of the 
    # covariance matrix is better conditioned by learning independence relations from the 
    # data. This is known as covariance selection.
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    group_lasso.fit(features)
    precision = group_lasso.precision_
    return precision


def get_mean_tied_precision(list_features, feature_list, num_classes, device):
    sample_class_mean = []
    out_count = 0
    num_output = len(feature_list)
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse 
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)           
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        print(temp_precision.shape)
        precision.append(temp_precision)
        
    return sample_class_mean, precision

def calc_class_mean_tied_precision(list_features, model, 
                                   num_classes, 
                                   feature_list, 
                                   train_loader, 
                                   device):
    '''
    function to compute class wise sample mean and tied precision (inverse of covariance)
    Authors: Mahalanobis Paper
    '''

    # RESPONSE: list_features is num_layers X num_classes X tensor ( num_data_points x feature_length_for_each_layer)
    #list_features, _ = getFeatures.get_all_features(model, num_classes, feature_list, train_loader, device)
    sample_class_mean, precision = get_mean_tied_precision(list_features, feature_list, num_classes, device)        
            
    return sample_class_mean, precision

def get_class_mean_class_precision(list_features, feature_list, num_classes, device):
    sample_class_mean = []
    out_count = 0
    num_output = len(feature_list)
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    # class_wise_precision is the list for storing class wise precision for different layers
    # it will be a list of list- for each layer, precision for each class
    class_wise_precision = []
    for k in range(num_output):
        class_wise_X = []
        for i in range(num_classes):
            class_wise_X.append(list_features[k][i] - sample_class_mean[k][i])
                
        # calculate class wise precision for the layer k
        class_wise_precision_for_layer_k = []
        for i in range(num_classes):
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(class_wise_X[i].cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            class_wise_precision_for_layer_k.append(temp_precision)
        class_wise_precision.append(class_wise_precision_for_layer_k)
    
    return sample_class_mean, class_wise_precision

def calc_class_mean_class_precision(list_features,
                                    model, 
                                    num_classes, 
                                    feature_list, 
                                    train_loader, 
                                    device):
    '''
    function to  compute class wise sample mean and class wise precision (inverse of covariance)
    Modified from original code calc_class_mean_tied_precision
    '''

    
    #list_features,_ = getFeatures.get_all_features(model, num_classes, feature_list, train_loader, device)
    sample_class_mean, class_wise_precision = get_class_mean_class_precision(list_features, feature_list, num_classes, device)
    
    return sample_class_mean, class_wise_precision



# ASK - is this useful for any of our methods now?
# RESPONSE - instead of using original features, use std of features
def modify_features_using_knn(features, 
                              knn_search,  
                              knn_args):
    '''
    Find nearest neighbors for input samples and create new features according to kwargs
    '''

    k = knn_args['k']
    modified_features = [[] for _ in range(len(features))]
    
    if type(features) == torch.Tensor:
        device = features.device
    else:
        device = None

    # iterate over features of this class
    for i,feat in enumerate(features):
        if device:
            neighbors = knn_search.predict(feat.detach().cpu().numpy(), k)
        else:
            neighbors = knn_search.predict(feat, k)

        # keep original feature in modified feature
        if knn_args['keep_original']:
            modified_features[i].append(feat)

        # keep mean of nearest neighbors in modified feature
        if knn_args['keep_knn_mean']:         
            knn_mean = np.mean(knn_search.org_features[neighbors], 0)
            if device:
                knn_mean = torch.from_numpy(knn_search).to(device)
            modified_features[i].append(knn_mean)
        
        #keep std of nearest neighbors in modified feature
        if knn_args['keep_knn_std']:
            # group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            # group_lasso.fit(knn_search.org_features[neighbors])
            # knn_covariance = group_lasso.covariance_
            # knn_covariance_diagonal = knn_covariance.diagonal()
            # knn_std_diagonal = np.sqrt(knn_covariance_diagonal).astype(np.float32)
            
            # RESPONSE: don't just look at neighbors, consider the self features
            #knn_std_diagonal = np.std(knn_search.org_features[neighbors], axis = 0)
            
            knn_features = knn_search.org_features[neighbors]            
            if device:                
                feat = feat.detach().cpu().numpy().reshape(1,-1)            
            #knn_with_original_features = np.vstack((feat,knn_features))            
            #knn_std_diagonal = np.std(knn_with_original_features, axis = 0)

            # FIXED to compute std after centering on feat (orig feature)
            knn_std_diagonal = np.sqrt(np.sum(np.square(knn_features-feat),axis=0))


            if device:
                knn_std_diagonal = torch.from_numpy(knn_std_diagonal).to(device)
            modified_features[i].append(knn_std_diagonal)

        if device:
            modified_features[i] = torch.cat(modified_features[i]).reshape(1,-1)
        else:
            modified_features[i] = np.concatenate(modified_features[i]).reshape(1,-1)

    if device:
        modified_features = torch.cat(modified_features,dim=0)
    else:
        modified_features = np.concatenate(modified_features,axis=0)
    
    return modified_features

def calc_knn_mean_precision(list_features,
                            model,
                            num_classes,
                            feature_list,
                            train_loader,
                            device,
                            cov_type,
                            knn_type_args,
                            knn_args):

    all_knn_search = []

    #list_features, _ = getFeatures.get_all_features(model, num_classes, feature_list, train_loader, device)
    list_features_modified = [[] for _ in range(len(list_features))]
    # iterating on features in each layer
    for l_idx,features in enumerate(list_features):

        for idx,features_class in enumerate(features):
            features[idx] = features_class.detach().cpu().numpy()

        # Create object for K-NN search algorithm
        knn_search = KNNSearch(np.concatenate(features,axis=0),knn_type_args)
        knn_search.fit()
        all_knn_search.append(knn_search)

        # modified features for each class 
        for features_class in features:     
            list_features_modified[l_idx].append(torch.from_numpy(modify_features_using_knn(features_class, knn_search, knn_args)).to(device))

    if cov_type =='tied_cov':
        sample_class_mean, precision = get_mean_tied_precision(list_features_modified, feature_list, num_classes, device) 
    elif cov_type =='class_cov':
        sample_class_mean, precision = get_class_mean_class_precision(list_features_modified, feature_list, num_classes, device)


    return all_knn_search, sample_class_mean, precision


def get_pca(list_features, model, num_classes, feature_list, train_loader, device):
    """
    return: pca_list: list of class-wise precision objects
    """
    import sklearn.covariance
    
    model.eval()
    
    num_output = len(feature_list)
    '''
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    
    for data, target in train_loader:
        data = data.to(device)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1

    '''
    # pca_list[k][i] with contain PCA object for kth layer and ith class
    pca_list = []
    for k in range(num_output):
        class_wise_pca = []
        for i in range(num_classes):
            pca = PCA(n_components=None)
            pca.fit(list_features[k][i].detach().cpu().numpy())
            class_wise_pca.append(pca)
        pca_list.append(class_wise_pca)
    
    return pca_list
