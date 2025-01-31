import os
import numpy as np
import pandas as pd
import numpy as np
import networkx as nx
import torch
from scipy.spatial import distance_matrix
from torch_geometric.data import HeteroData


def remove_duplicates(dict_list):
    """
    Remove duplicate dictionary elements from a list.
    """
    seen = set()
    unique_list = []
    for d in dict_list:
        # Use frozenset as the uniqueness criterion
        d_tuple = frozenset((k, tuple(v) if isinstance(v, np.ndarray) else v) for k, v in d.items())
        if d_tuple not in seen:
            seen.add(d_tuple)
            unique_list.append(d)
    return unique_list


def build_graph_from_list(node_list, k_nearest=8):
    """
    Construct a graph from a list of nodes and create edges using the k-nearest neighbors.

    Parameters:
    - node_list: A list where each element is a dictionary containing 'feature', 'label', and 'pos' keys.
    - k_nearest: Number of nearest neighbors each node should connect to.

    Returns:
    - edge_index: Edge index of the graph in PyTorch format.
    - edge_weights: Edge weights.
    - node_features: Node feature matrix.
    - node_labels: Node labels.
    """
    # Extract coordinates, features, and labels
    coords = np.array([node['pos'] for node in node_list])
    features = np.array([node['feature'] for node in node_list])
    labels = np.array([node['label'] for node in node_list])

    # Compute the distance matrix
    dist_matrix = distance_matrix(coords, coords)

    # Construct a NetworkX graph
    G = nx.Graph()
    for k in range(len(node_list)):
        G.add_node(k, pos=coords[k], feature=features[k], label=labels[k])

    # Add edges by selecting the k-nearest neighbors
    for k in range(len(node_list)):
        # Sort all distances for node k and get the nearest k neighbors (excluding itself)
        nearest_indices = np.argsort(dist_matrix[k])[1:k_nearest + 1]
        for j in nearest_indices:
            G.add_edge(k, j, weight=1 / (dist_matrix[k, j] + 1e-6))  # Weight is the inverse of distance

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_weights = torch.tensor([G[k][j]['weight'] for k, j in G.edges], dtype=torch.float)
    node_features = torch.tensor([G.nodes[k]['feature'] for k in G.nodes], dtype=torch.float)
    node_labels = torch.tensor([G.nodes[k]['label'] for k in G.nodes], dtype=torch.float)

    return edge_index, edge_weights, node_features, node_labels


dataset = ['Ours', 'CRC']
levels = ['08', '16']

for data in dataset:
    for level in levels:
        csv_infor_bins = f'./Our_HD_data/{data}/data_infor/{level}'
        for file in os.listdir(csv_infor_bins):
            graph_224_infor = []
            graph_512_infor = []
            tmp_level_new_infor = []

            tmp_name = file[:-4]

            # Load patch information for 224x224 resolution
            csv_infor_224 = pd.read_csv(
                f'.Our_HD_data/{data}/16_information/csv_infor_224/' + f'{tmp_name}_patches_224.csv')
            # Load gene expression data for 224x224 resolution
            gene_expression_224 = pd.read_csv(
                f'./Our_HD_data/{data}/16_information/gene_expression_224/select_{level}/' + f'{tmp_name}.csv', header=None)
            # Load index mapping for 224x224
            index_224 = np.load(
                f'./Our_HD_data/{data}/16_information/index_name_224/' + f'{tmp_name}.npy',
                allow_pickle=True)
            # Load extracted features for 224x224
            feature_224 = np.load(
                f'./Our_HD_data/{data}/extracted_feature/224/' + f'{tmp_name}.npy',
                allow_pickle=True).item()

            # Load patch information for 512x512 resolution
            csv_infor_512 = pd.read_csv(
                f'./Our_HD_data/{data}/16_information/csv_infor_512/' + f'{tmp_name}_patches_512.csv')
            # Load gene expression data for 512x512 resolution
            gene_expression_512 = pd.read_csv(
                f'./Our_HD_data/{data}/16_information/gene_expression_224/select_{level}/' + f'{tmp_name}.csv', header=None)
            # Load index mapping for 512x512
            index_512 = np.load(
                f'./Our_HD_data/{data}/16_information/index_name_512/' + f'{tmp_name}.npy',
                allow_pickle=True)
            # Load extracted features for 512x512
            feature_512 = np.load(
                f'./Our_HD_data/{data}/extracted_feature/512/' + f'{tmp_name}.npy',
                allow_pickle=True).item()

            # Load dataset information
            csv_infor_bins = np.load(
                f'./Our_HD_data/{data}/data_infor/{level}/' + f'{tmp_name}.npy',
                allow_pickle=True)

            # Paths to save the processed data
            tmp_new_infor_save_path = f'./Our_HD_data/{data}/our_data_infor/{level}/' + f'{tmp_name}.npy'
            tmp_new_pt_save_path = f'./Our_HD_data/{data}/our_data_pt/{level}/' + f'{tmp_name}.pt'

            for i in range(csv_infor_bins.shape[0]):
                tmp_old_infor = csv_infor_bins[i]

                # Process 224x224 data
                tmp_224_infor = index_224[i]
                tmp_224_idx = tmp_224_infor['row_index']
                tmp_224_name = tmp_224_infor['first_column_value'][:-4]
                tmp_224 = csv_infor_224.iloc[tmp_224_idx]
                tmp_224_expression = gene_expression_224.iloc[:, tmp_224_idx]
                print(tmp_224_expression.shape)
                try:
                    tmp_224_feature = feature_224[tmp_224_name]
                except KeyError:
                    print(f"Key '{tmp_224_name}' not found, skipping...")
                    continue

                tmp_224_locations = (tmp_224['i'], tmp_224['j'])

                # Process 512x512 data
                tmp_512_infor = index_512[i]
                tmp_512_idx = tmp_512_infor['row_index']
                tmp_512_name = tmp_512_infor['first_column_value'][:-4]
                tmp_512 = csv_infor_512.iloc[tmp_512_idx]
                tmp_512_expression = gene_expression_512.iloc[:, tmp_512_idx]
                print(tmp_512_expression.shape)
                tmp_512_feature = feature_512[tmp_512_name]
                tmp_512_locations = (tmp_512['i'], tmp_512['j'])

                # Create a dictionary with all extracted information
                tmp_new_infor = {'img_path': tmp_old_infor['barcode'], 'label': tmp_old_infor['expression'],
                                 'position': tmp_old_infor['pos'], 'feature_1024': tmp_224_feature,
                                 'feature_512': tmp_512_feature, 'label_1024': np.array(tmp_224_expression),
                                 'label_512': np.array(tmp_512_expression)}
                tmp_level_new_infor.append(tmp_new_infor)

                graph_224_infor.append(
                    {'feature': tmp_224_feature, 'label': np.array(tmp_224_expression), 'pos': tmp_224_locations})
                graph_512_infor.append(
                    {'feature': tmp_512_feature, 'label': np.array(tmp_512_expression), 'pos': tmp_512_locations})

            np.save(tmp_new_infor_save_path, tmp_level_new_infor)
            print(f'Data info saved to {tmp_new_infor_save_path}')

            # Create a heterogeneous graph
            pt_data = HeteroData()
            layer2_edge_index, layer2_edge_weights, layer2_features, layer2_labels = build_graph_from_list(remove_duplicates(graph_512_infor), k_nearest=8)
            layer3_edge_index, layer3_edge_weights, layer3_features, layer3_labels = build_graph_from_list(remove_duplicates(graph_224_infor), k_nearest=8)

            # Assign features, edges, and labels to each graph layer
            pt_data['layer_512'].x = layer2_features
            pt_data['layer_1024'].x = layer3_features

            save_path = tmp_new_pt_save_path
            torch.save(pt_data, save_path)
            print(f"HeteroData has been saved as: {save_path}")
