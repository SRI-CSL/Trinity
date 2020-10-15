from __future__ import print_function

import numpy as np
import torch


import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import random
random.seed(20)


# RESPONSE: To dump features in a file 
def dump_features_for_given_layer(train_loader, 
                                  model, 
                                  layer_index, 
                                  out_file, 
                                  device):
    '''
    function for dumping dataset's features got from the layer with index 'layer_index' 
    in the model the features will be dumped in the output file out_file.
    '''
    feature_list = [] 
    label_list   = []
        
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        # fetch features of the data from the intermediate layer of the model
        features = model.intermediate_forward(data, layer_index)
        features = features.view(features.size(0), features.size(1), -1)
        features = torch.mean(features.data, 2)
        labels   = labels.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        label_list.append(labels)
        feature_list.append(features)
        
    print("features len: ", len(feature_list))
    all_labels = np.concatenate(label_list).flatten()
    all_features = np.concatenate(feature_list)
    np.savez(out_file, labels = all_labels, features = all_features) 


# ASK: get_all_features uses model.feature_list instead of intermediate_forward 
# used in dump_features - why?
def get_all_features(model, 
                    num_classes, 
                    feature_list, 
                    layer_list,
                    train_loader, 
                    device):
    model.eval()
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    labels = []
    for data, target in train_loader:
        labels.append(target.cpu().numpy())
        # total += data.size(0)
        data = data.to(device)
        output, out_features = model.feature_list(data, layer_list)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2) # avgpool2d and mean would yield same - taking avgpool2d 
            
        # compute the accuracy
        # pred = output.data.max(1)[1]
        # equal_flag = pred.eq(target.to(device)).cpu()
        # correct += equal_flag.sum()
        
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
        
    # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    #print("debug",(list_features[0][0].shape))
    #print(len(list_features[0]))
    #print(len(list_features))
    # RESPONSE: list_features is num_layers X num_classes X tensor ( num_data_points x feature_length_for_each_layer)
    return list_features,np.concatenate(labels)



# RESPONSE: not used - for plotting 
def plot_TSNE(n_components, 
              verbose, 
              perplexity, 
              n_iter, 
              random_state,
              color_list, 
              class_list, 
              feature_file_list, 
              plot_file_name):
    '''
    function to plot the TSNE map to visualize the features in low-dimensional space
    '''
    print("random_state: ", random_state)
    for i,f in enumerate(feature_file_list):
        input_features_file_data = np.load(f)
        if i == 0:
            print("i- ", i)
            features = input_features_file_data['features']
            labels = input_features_file_data['labels']
        else:
            print("i- ", i)
            features = np.vstack([features, input_features_file_data['features']])
            ood_labels = np.ndarray(len(input_features_file_data['features']))
            ood_labels[:] = 11
            labels = np.concatenate([labels, ood_labels]).flatten()

    # ----------------- T-SNE -------------------
    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(features)
    print ("tsne_results shape: ", tsne_results.shape)

    plt.figure(figsize=(5,4))

    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    colours = ListedColormap(color_list)
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap=colours)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_list,loc='lower left',
           fontsize=5)
    plt.title("T-SNE")
    plt.savefig('{}.pdf'.format(plot_file_name))

    # --------- T-SNE on K-means clustering --------
    # kmeans = KMeans(n_clusters=len(class_list), random_state=0, max_iter=1000).fit(features)
    # plt.figure(figsize=(5,4))
    # scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=kmeans.labels_, cmap=colours)
    # plt.legend(handles=scatter.legend_elements()[0], labels=class_list)
    # plt.title("T-SNE based on K-means clustering")
    # plt.savefig('T-SNE based on K-means clustering')
