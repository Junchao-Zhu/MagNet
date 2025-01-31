import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models.graph_construction import calcADJ
import os


def find_closest_column(target_sequence, tensor):
    target_sequence = torch.tensor(target_sequence)
    distances = torch.norm(tensor - target_sequence, dim=1)
    closest_index = torch.argmin(distances)

    return closest_index.item()


class ImageGraphDataset(Dataset):
    def __init__(self, data_infor_path, graph_path, transform=None):
        # Load data_list and graph information from the.npy file
        self.data_list = np.load(data_infor_path, allow_pickle=True).tolist()
        self.graph = torch.load(graph_path)
        self.transform = transform
        self.name = data_infor_path.split('/')[-2]
        self.layer_512 = self.graph['layer_512'].y
        self.layer_224 = self.graph['layer_224'].y

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_dir = './Our_HD_data/Ours'
        image_path = os.path.join(img_dir, self.data_list[idx]['img_path'])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data_list[idx]['label']
        position = self.data_list[idx]['position']
        feature_224 = self.data_list[idx]['feature_224']
        feature_512 = self.data_list[idx]['feature_512']

        idx_224 = find_closest_column(self.data_list[idx]['label_224'], self.layer_1024)
        idx_512 = find_closest_column(self.data_list[idx]['label_512'], self.layer_512)

        return image, label.astype(np.float32), position, int(idx_224), int(idx_512), feature_224.astype(
            np.float32), feature_512.astype(np.float32),


def collate_fn(batch, graph):
    # Extract images, labels, locations, and other features
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512 = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    positions = torch.tensor(positions)
    n_adj = min(positions.shape[0] - 1, 8)

    adj_sub = calcADJ(positions.numpy(), k=n_adj)

    features_1024 = torch.tensor(features_1024)
    features_512 = torch.tensor(features_512)
    idx_1024 = torch.tensor(idx_1024_list)
    idx_512 = torch.tensor(idx_512_list)

    return images, labels, adj_sub, positions, idx_1024, idx_512, features_1024, features_512, graph


# Function: Create a separate DataLoader for each WSI
def create_dataloaders_for_each_file(npy_file_paths, batch_size=56, transform=None):
    dataloaders = {}
    for npy_file in npy_file_paths:
        name = npy_file.split('/')[-1][:-4]

        graph_path = f'./Our_HD_data/Ours/our_data_pt/16/{name}.pt'

        dataset = ImageGraphDataset(npy_file, graph_path, transform=transform)
        graph_data = dataset.graph

        def collate_fn_with_params(batch, graph_datas=graph_data):
            return collate_fn(batch, graph_datas)

        dataloaders[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn_with_params)
    return dataloaders
