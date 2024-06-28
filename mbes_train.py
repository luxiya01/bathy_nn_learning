import argparse
import copy
import logging
import os.path as osp
from datetime import datetime
import sys

import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from dataset import PairDataset, SubmapDataset, SubmapTransform
from mbes_data.lib.utils import load_config
from models import *
from param import *
from mbes_data.datasets.mbes_data import get_multibeam_datasets
from easydict import EasyDict as edict
from tqdm import tqdm

LOAD_PATH = 'sin0627_19.pt' #None  # if no old model

def train_and_valid(epoch):

    ## ==============
    ## train
    model.train()

    training_loss = 0
    count_batch, count_triplets = 0, 0
    bad_batch = 0

    for idx, data in tqdm(enumerate(train_loader, len(train_loader))):
        if data is None:
            logging.info(f"No data in batch {idx}!")
            bad_batch += 1
            continue

        for item in data:
            item = item.to(device)
        optimizer.zero_grad()

        out = model(data[0], data[1], data[2])
        
        # No triplets
        if type(out) is not tuple:
            bad_batch += 1 
            continue

        (out_anc, out_pos, out_neg), score_anc, _ = out

        pos_dist = F.pairwise_distance(out_anc, out_pos)
        neg_dist = F.pairwise_distance(out_anc, out_neg)

        loss = triplet_loss(pos_dist - neg_dist + CONFIG["Margin"]).mul(score_anc.ravel()).sum()
        training_loss += loss.detach().cpu().numpy()

        loss.backward()
        optimizer.step()
        count_batch += 1
        count_triplets += out_anc.shape[0]
    
    # FIXME: what if no triplets in all batches?
    if count_batch != 0:
        training_loss = training_loss / count_batch #len(train_loader)  # mean over samples, count_batch
        num_triplets_per_map = count_triplets / count_batch / BATCH_SIZE
    else:
        training_loss = -1
        num_triplets_per_map = -1
        print("all bad batches!")
            
    print(f"Epoch {epoch}, Training loss: ", training_loss)

    training_log = {
     "training_loss": training_loss, "num_triplets_per_map ": num_triplets_per_map,
     }

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if epoch % SAVE_EPOCH == 0 or epoch == START_EPOCH + NUM_EPOCH - 1:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': training_loss,
                }, osp.join(osp.dirname(osp.realpath(__file__))
                , PATH_CKPT, f"{dataset_name}_{current_time}_epoch_{epoch}.pt")
                )
    
    ## ===================
    ## test (validation)
    test_loss = 0  # triplet loss
    easy = 0
    hard = 0
    bad_batch = 0
    model.eval()

    for data in tqdm(test_loader):
        for item in data:
            item = item.to(device)
        
        with torch.no_grad():

            out = model(data[0], data[1], data[2])
            
            # No triplets
            if type(out) is not tuple: 
                bad_batch += 1
                continue

            (out_anc, out_pos, out_neg), score_anc, _ = out

            pos_dist = F.pairwise_distance(out_anc, out_pos)
            neg_dist = F.pairwise_distance(out_anc, out_neg)

            loss = triplet_loss(pos_dist - neg_dist + CONFIG["Margin"]).mul(score_anc.ravel()).sum()

        test_loss += loss.detach().cpu().numpy()
        dist_easy = (neg_dist - pos_dist - CONFIG["Margin"] > 0) * 1.0
        dist_hard = (neg_dist < pos_dist) * 1.0
        easy += dist_easy.mean().item()
        hard += dist_hard.mean().item()

    if len(test_loader) != bad_batch:
        test_loss = test_loss / (len(test_loader) - bad_batch)
        test_easy = easy / (len(test_loader) - bad_batch)
        test_hard = hard / (len(test_loader) - bad_batch)
        test_semi = 1 - test_easy - test_hard
    else:
        test_loss, test_easy, test_hard, test_semi = -1, -1, -1, -1

    print(f"Epoch {epoch}, Validation triplet loss: ", test_loss)
    print(f"Epoch {epoch}, easy triplets: ", test_easy)
    print(f"Epoch {epoch}, semi-hard triplets: ", test_semi)
    print(f"Epoch {epoch}, hard triplets: ", test_hard)

    validation_log = {"validation_triplet_loss": test_loss, "easy triplets": test_easy,
     "hard triplets": test_hard, "semi-hard triplets": test_semi}
    last_lr = scheduler.get_last_lr()[0]
    learning_rate = {"learning_rate": last_lr}
    print(f"Epoch {epoch}, learning_rate: ", last_lr)
    training_log.update(validation_log)
    training_log.update(learning_rate)
    wandb.log(training_log, step=epoch)

    scheduler.step()
    #===========

# ========= main ==========
if __name__ == '__main__':
    # Set up logging
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        handlers=[ch])

    logging.basicConfig(level=logging.INFO, format="")

    # Load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--mbes_config',
                        type=str,
                        default='mbes_data/configs/mbesdata_test_meters.yaml',
                        help='Path to multibeam data config file')
    parser.add_argument('--network_config',
                        type=str,
                        default='network_configs/kpthresh07.yaml',
                        help='Path to network config file')
    args = parser.parse_args()
    mbes_config = edict(load_config(args.mbes_config))
    network_config = edict(load_config(args.network_config))
    print(f'MBES data config: {mbes_config}')
    print(f'Network config: {network_config}')

    config = copy.deepcopy(mbes_config)
    for k, v in network_config.items():
        if k not in config:
            config[k] = v
    if config.use_gpu:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    ### ==============
    ### main
    ### ==============
    dataset_name = 'MBES'
    project_name = f'BathyNN-{dataset_name}'
    path = f'{DATA_PATH}{dataset_name}'

    wandb.init(project=project_name)
    config.batch_size = BATCH_SIZE
    now = datetime.now()
    config.local_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    config.dataset_type = 'multibeam_npy_for_bathynn'
    train_set, val_set, test_set = get_multibeam_datasets(config)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    triplet_loss = torch.nn.ReLU()
    model = Matcher().to(device)
        
    START_EPOCH = 0
    if LOAD_PATH:
        old_dict = torch.load(LOAD_PATH)
        model.load_state_dict(old_dict['model_state_dict'])
        START_EPOCH = old_dict['epoch'] + 1
        print(f"Resume training from epoch {START_EPOCH}" )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    print("Begin training...")
    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCH):
        train_and_valid(epoch)
    print("Stop training...")