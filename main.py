import torch
from dataloader import ImageGraphDataset, collate_fn, create_dataloaders_for_each_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.graph_construction import edge_index_to_adj_matrix
from models.model import MagNet
import warnings
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
import torchvision
import logging
import sys
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import time
from models.losses import calculate_pcc, pcc_loss
from cross_validation import four_fold_cross_validation
import pandas as pd


warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Our_HD_data/Ours',
                    help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=25000, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float, default=0.000015, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=196, help='repeat')
parser.add_argument('--level', type=str, default='16', help='Resolution')

args = parser.parse_args()

root = args.root_path
label_root = os.path.join(root, f'our_data_infor/{args.level}')
image_folder_paths = sorted([f for f in os.listdir(label_root) if f.endswith('.npy')])
data_name = args.root_path.split('/')[-1]

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
max_iterations = args.max_iterations
base_lr = args.base_lr
lr_decay = args.lr_decay
loss_record = 0
batch_size = args.batch_size
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    save_path = f'./weight/ours_{args.level}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    img_transform = transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.RandomVerticalFlip(),
                                        torchvision.transforms.RandomApply(
                                            [torchvision.transforms.RandomRotation((90, 90))]),
                                        torchvision.transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    mse_loss_fn = nn.MSELoss()

    cross_mse, cross_mae, cross_pcc_patch, = [], [], []

    image_dict = {}
    fold_splits = four_fold_cross_validation(label_root)
    mse_metrics = []
    mae_metrics = []
    pcc_metrics = []
    save_dict = {}

    # Perform four-fold cross-validation
    for fold_index, (train_labels_path, val_labels_path) in enumerate(fold_splits):
        model = MagNet().cuda()
        print(f"\nFold {fold_index + 1}:")
        print((train_labels_path, val_labels_path))

        val_labels = []
        train_labels = []

        train_dataloaders = create_dataloaders_for_each_file(train_labels_path, batch_size=256, transform=img_transform)
        test_dataloader = create_dataloaders_for_each_file(val_labels_path, batch_size=256, transform=img_transform)

        iter_num = 0
        max_epoch = 51
        lr_ = base_lr

        writer = SummaryWriter(save_path + '/log')

        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        model.cuda()

        for epoch_num in tqdm(range(max_epoch), ncols=70):
            model.train()

            file_names = list(train_dataloaders.keys())
            random.shuffle(file_names)

            for file_name in file_names:
                tmp_loader = train_dataloaders[file_name]

                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * (1 - float(iter_num) / max_iterations) ** lr_decay

                print(f"Processing file: {file_name}")
                tmp_name = file_name.split('/')[-1][:-4]

                for images, labels, adj_sub, positions, idx_224, idx_512, features_224, features_512, graph_torch in tmp_loader:
                    graph_512 = graph_torch['layer_512']
                    graph_224 = graph_torch['layer_224']

                    batch_data = [
                        images, adj_sub, features_512, features_224,
                        graph_512.x, graph_224.x,
                        edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_224),
                        idx_224, idx_512
                    ]

                    # Calculate hybrid loss functions
                    batch_data = [data.to(device) for data in batch_data]

                    pred, pred_512, pred_224 = model(*batch_data)

                    pred_512_con = pred_512.cpu()[idx_512.cpu()]
                    pred_224_con = pred_224.cpu()[idx_224.cpu()]

                    loss_other_layer = pcc_loss(graph_512.y.cuda(), pred_512) + pcc_loss(graph_224.y.cuda(), pred_224)

                    loss_consistency = pcc_loss(pred_512_con.cuda(), labels.cuda()) + pcc_loss(pred_224_con.cuda(), labels.cuda())

                    loss = 0.75*mse_loss_fn(pred, labels.cuda()) + 0.1*pcc_loss(pred, labels.cuda()) + 0.05 * loss_other_layer + 0.05 * loss_consistency

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iter_num = iter_num + 1
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/loss', loss, iter_num)

                    logging.info(
                        'iteration %d :loss: %5f, mse loss: %5f, lr: %f5' %
                        (iter_num, loss, mse_loss_fn(pred, labels.cuda()),
                         optimizer.param_groups[0]['lr']))

                    if iter_num >= max_iterations:
                        break
                    time1 = time.time()

            if epoch_num % 10 == 0:
                model.eval()
                val_loss = 0.0
                val_pcc = 0.0
                val_mae = 0.0
                total_samples = 0

                best_mse = 10000
                mae = 0

                with torch.no_grad():
                    for file_name, tmp_loader in test_dataloader.items():
                        print(f"Processing file: {file_name}")
                        tmp_name = file_name.split('/')[-1][:-4]
                        tmp_total = 0

                        for images, labels, adj_sub, positions, idx_224, idx_512, features_224, features_512, graph_torch in tmp_loader:
                            graph_512 = graph_torch['layer_512']
                            graph_224 = graph_torch['layer_224']

                            batch_data = [
                                images, adj_sub, features_512, features_224,
                                graph_512.x, graph_224.x,
                                edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_224),
                                idx_224, idx_512
                            ]
                            batch_data = [data.to(device) for data in batch_data]

                            pred, b, c, = model(*batch_data)

                            # Calculate related metrics
                            loss = mse_loss_fn(pred, labels.cuda())
                            val_loss += loss.item() * images.size(0)
                            total_samples += images.size(0)

                            mae += np.mean(np.abs(pred.cpu().numpy() - labels.cpu().numpy())) * images.size(0)
                            val_pcc += calculate_pcc(pred, labels.cuda()) * images.size(0)
                            tmp_total += images.size(0)
                            print(f'mse {loss}, pcc {calculate_pcc(pred, labels.cuda())}')

                torch.save(model.state_dict(), os.path.join(save_path, f"model_best_{fold_index}.pth"))
                print(f'Best model saved to {os.path.join(save_path, f"model_best_{fold_index}.pth")}')

                # Show results
                print(f"Validation MSE Loss: {val_loss / total_samples}")
                print(f"Mean Absolute Error (MAE) with NumPy:{mae / total_samples}")
                print(f"Validation mean pcc per patch: {val_pcc / total_samples}")

            if epoch_num == max_epoch-1:
                model.eval()

                with torch.no_grad():
                    for file_name, tmp_loader in test_dataloader.items():
                        print(f"Processing file: {file_name}")
                        tmp_name = file_name.split('/')[-1][:-4]
                        tmp_total = 0
                        mae = 0
                        mean_pcc_per_patch = 0
                        val_loss = 0
                        total_samples = 0

                        for images, labels, adj_sub, positions, idx_224, idx_512, features_224, features_512, graph_torch in tmp_loader:
                            graph_512 = graph_torch['layer_512']
                            graph_224 = graph_torch['layer_224']

                            batch_data = [
                                images, adj_sub, features_512, features_224,
                                graph_512.x, graph_224.x,
                                edge_index_to_adj_matrix(graph_512), edge_index_to_adj_matrix(graph_224),
                                idx_224, idx_512
                            ]

                            batch_data = [data.to(device) for data in batch_data]
                            pred, b, c, = model(*batch_data)

                            loss = mse_loss_fn(pred, labels.cuda())
                            val_loss += loss.item() * images.size(0)
                            total_samples += images.size(0)
                            mean_pcc_per_patch += (calculate_pcc(pred, labels.cuda())) * images.size(0)
                            mae += np.mean(np.abs(pred.cpu().numpy() - labels.cpu().numpy())) * images.size(0)

                        mse_metrics.append(val_loss / total_samples)
                        mae_metrics.append(mae / total_samples)
                        pcc_metrics.append(mean_pcc_per_patch.cpu().numpy() / total_samples)

        save_dict['mse'] = mse_metrics
        save_dict['mae'] = mae_metrics
        save_dict['pcc'] = pcc_metrics
        print(save_dict)

        df = pd.DataFrame(save_dict)
        output_file = os.path.join(save_path, 'ours_metrics.xlsx')
        df.to_excel(output_file, index=False)

        print(f"Results have saved to {output_file}")
