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

import random
random.seed(20)




class KNNSearch:
    def __init__(self,features,kwargs):

        self.org_features = features
        if kwargs["normalize"]:
            self.features  = preprocessing.normalize(features,norm='l2')
        else:
            self.features  = features

        self.kwargs    = kwargs
        self.predictor = None
    
    def fit(self):
        if self.kwargs['algorithm'] =='datasketch':
            self.__datasketch_fit()
        elif self.kwargs['algorithm']=='annoy':
            self.__annoy_fit()
        elif self.kwargs['algorithm']=='exact':
            self.__exhaustive_fit()
        elif self.kwargs['algorithm']=='falconn':
            self.__falconn_fit()
        elif self.kwargs['algorithm']=='descent':
            self.__descent_fit()
        elif self.kwargs['algorithm']=='random':
            self.__random_fit()
        else:
            raise Exception("Algorithm=[{}] not yet implemented".format(self.kwargs['algorithm']))

    def predict(self,input,k):
        if self.kwargs['algorithm'] =='datasketch':
            return self.__datasketch_predict(input,k)
        elif self.kwargs['algorithm']=='annoy':
            return self.__annoy_predict(input,k)
        elif self.kwargs['algorithm']=='exact':
            return self.__exhaustive_predict(input,k)
        elif self.kwargs['algorithm']=='falconn':
            return self.__falconn_predict(input,k)
        elif self.kwargs['algorithm']=='descent':
            return self.__descent_predict(input,k)
        elif self.kwargs['algorithm']=='random':
            return self.__random_predict(input,k)
        else:
            raise Exception("Algorithm=[{}] not yet implemented".format(self.kwargs['algorithm']))

    def __datasketch_fit(self):
        if self.kwargs['create']:
            # Create a list of MinHash objects
            min_hash_obj_list = []
            forest = MinHashLSHForest(num_perm=self.kwargs['num_perm'])
            for i in range(len(self.features)):
                min_hash_obj_list.append(MinHash(num_perm=self.kwargs['num_perm']))
                for d in self.features[i]:
                    min_hash_obj_list[i].update(d)
                forest.add(i, min_hash_obj_list[i])
            # IMPORTANT: must call index() otherwise the keys won't be searchable
            forest.index()   
            with open(self.kwargs['file_path'],"wb") as f:
                pickle.dump(forest, f)
                pickle.dump(min_hash_obj_list, f)
            self.predictor = [forest,min_hash_obj_list]
        else:
            with open(self.kwargs['file_path'], "rb") as f:
                forest = pickle.load(f)
                min_hash_obj_list = pickle.load(f)
                self.predictor = [forest,min_hash_obj_list]

    def __datasketch_predict(self,input,k):
        forest,min_hash_obj_list = self.predictor
        if type(input)==int:
            return forest.query(min_hash_obj_list[input], k)
        else:
            min_hash_obj = MinHash(num_perm=self.kwargs['num_perm'])
            for d in input:
                min_hash_obj.update(d)
            return forest.query(min_hash_obj, k)

    def __annoy_fit(self):
        if self.kwargs['create']:
            indexer = AnnoyIndex(self.features.shape[1],self.kwargs['metric'])
            for i,f in enumerate(self.features):
                indexer.add_item(i,f)
            indexer.build(self.kwargs['num_trees'])
            indexer.save(self.kwargs['file_path'])
            self.predictor = indexer
        else:
            forest = AnnoyIndex(self.features.shape[1], self.kwargs['metric'])
            forest.load(self.kwargs['file_path'])
            self.predictor = forest
    
    def __annoy_predict(self,input,k):
        annoy_forest = self.predictor
        if type(input)==int:
            return annoy_forest.get_nns_by_item(input, k, search_k=-1, include_distances=False)
        else:
            return annoy_forest.get_nns_by_vector(input, k, search_k=-1, include_distances=False)

    def __exhaustive_fit(self):
        self.predictor = NearestNeighbors(algorithm='ball_tree')
        self.predictor.fit(self.features)
    
    def __exhaustive_predict(self,input,k):
        if type(input)==int:
            return self.predictor.kneighbors(self.features[input].reshape(1,-1),n_neighbors=k,return_distance=False)[0]
        else:
            return self.predictor.kneighbors(input.reshape(1,-1),n_neighbors=k,return_distance=False)[0]
    
    def __falconn_fit(self):
        """
        Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
        """

        import falconn

        dimension = self.features.shape[1]
        nb_tables = self.kwargs['nb_tables']
        number_bits = self.kwargs['number_bits']

        # LSH parameters
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = nb_tables
        params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
        params_cp.seed = 5721840
        # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        # we build number_bits-bit hashes so that each table has
        # 2^number_bits bins; a rule of thumb is to have the number
        # of bins be the same order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(number_bits, params_cp)
        self._falconn_table = falconn.LSHIndex(params_cp)
        self._falconn_query_object = None
        self._FALCONN_NB_TABLES = nb_tables

        # Center the dataset and the queries: this improves the performance of LSH quite a bit.
        self.center    = np.mean(self.features, axis=0)
        self.features -= self.center

        # add features to falconn table
        self._falconn_table.setup(self.features)

    def __falconn_predict(self,input,k):

        # Normalize input if you care about the cosine similarity
        if type(input)==int:
            input = self.features[input]
        else:
            if self.kwargs['normalize']:
                input /= np.linalg.norm(input)
                # Center the input and the queries: this improves the performance of LSH quite a bit.
                input -= self.center        

        # Late falconn query_object construction
        # Since I suppose there might be an error
        # if table.setup() will be called after
        if self._falconn_query_object is None:
            self._falconn_query_object = self._falconn_table.construct_query_object()
            self._falconn_query_object.set_num_probes(
                self._FALCONN_NB_TABLES
            )

        query_res = self._falconn_query_object.find_k_nearest_neighbors(input,k)
        return query_res

    def __descent_fit(self):
        self.predictor = NNDescent(data=self.features, metric=self.kwargs['metric'])
    
    def __descent_predict(self,input,k):
        input = np.expand_dims(input, axis=0) # input should be an array of search points
        index = self.predictor
        return index.query(input, k)[0][0] # returns indices of NN, distances of the NN from the input

    def __random_fit(self):
        pass
    
    def __random_predict(self,input,k):
        rand_index_list = []
        for i in range(k):
            rand_index_list.append(random.randint(0,len(self.features)-1))

        return rand_index_list