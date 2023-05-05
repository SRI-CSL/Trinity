from torch.utils.data import Dataset
from dgl.data.utils import load_graphs
from glob import glob
import dgl
import torch
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class GraphDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, path_list, feat_option = 'all_feat'):
        print ("Dataset size: ",len(path_list))
        self.path_list = path_list
        self.feat_option = feat_option
 
    def __getitem__(self, index):

        # G, _ = load_graphs(self.path_list[index])
        # G = G[0]
        # if self.feat_option == 'all_feat':
        #     all_feature = []
        #     for node_id in range(G.number_of_nodes()):
        #         all_feature.append(list(G.ndata['geo_feature'][node_id]) + list(G.ndata['vis_feature'][node_id]))
        #     G.ndata['feature'] = torch.tensor(all_feature) 

        return self.path_list[index]

    def __len__(self):
        return len(self.path_list)

def filter_graphs(graph_dir, node_num_thresh = 4):

    print (f"[[ Filtering graphs with <= {node_num_thresh} nodes ]]")
    graph_path_list = sorted(glob(graph_dir + '/*.bin'))
    original_stats = defaultdict(int)
    # remove small graphs
    graph_path_list_sel = []
    for g_path in tqdm(graph_path_list):
        G, _ = load_graphs(g_path)
        G = G[0]
        num_nodes = G.number_of_nodes()
        num_edges = G.num_edges()
        if num_nodes * (num_nodes - 1) != num_edges:
            print("Error #node * (#node - 1) != #edges",g_path)
            temp_g_path = Path(g_path)
            # temp_g_path.unlink()
        original_stats[num_nodes] +=1
        if num_nodes > node_num_thresh:
            graph_path_list_sel.append(g_path)

    print("Overall Stats:")
    for node_count in sorted(original_stats):
        print(original_stats[node_count]," images have ",node_count , "nodes")
    return graph_path_list_sel

def get_dataset(path_list = ''):
    dataset = GraphDataset(path_list = path_list)
    return dataset
