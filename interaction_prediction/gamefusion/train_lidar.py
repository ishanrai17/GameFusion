import torch
import csv
import argparse
import time
import os
import sys
import numpy as np
import logging
from torch import optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Setup paths
sys.path.insert(0, '/content/GameFusion')
sys.path.insert(1, '/content/GameFusion/interaction_prediction')
sys.argv = [a.replace('--local-rank', '--local_rank') for a in sys.argv]

# Import downstream utilities and your custom MAE
from utils.inter_pred_utils import *
from interaction_prediction.gamefusion.utilities import DrivingData
from model.LiDAR_CNN_MAE import HierarchicalLiDARCNNMAE  # Ensure this points to your MAE file


def training_epoch(train_data, model, optimizer, epoch, args):
    epoch_loss = []
    model.train()
    current = 0
    start_time = time.time()
    size = len(train_data)

    for batch in train_data:
        # batch[7] holds the saved .npz uint8 lidar_bev array: [B, 11, 24, 748, 748]
        lidar_sequence = batch[7].to(args.local_rank)
        
        B, T, C, H, W = lidar_sequence.shape
        
        # 1. Cast uint8 to float32 dynamically on the GPU to preserve CPU RAM bandwidth
        x = lidar_sequence.to(torch.float32)
        
        # 2. Fold Time into Batch to process frames as independent spatial snapshots
        # Shape becomes: [B * 11, 24, 748, 748]
        x_folded = x.view(B * T, C, H, W)

        optimizer.zero_grad()
        
        # The forward pass automatically handles dynamic padding (748->752), masking, and MSE loss
        loss = model(x_folded)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        current += B
        epoch_loss.append(loss.item())

        if dist.get_rank() == 0:
            logging.info(
                f"\rTrain Progress: [{current:>6d}/{size*args.batch_size:>6d}] " +
                f"| Reconstruction MSE Loss: {np.mean(epoch_loss):>.6f} | " +
                f"{(time.time()-start_time)/current:>.4f}s/sample"
            )
    
    return epoch_loss


def validation_epoch(valid_data, model, epoch, args):
    model.eval()
    current = 0
    start_time = time.time()
    size = len(valid_data)
    epoch_loss = []

    if dist.get_rank() == 0:
        logging.info(f'Validation... Epoch {epoch+1}')

    for batch in valid_data:
        lidar_sequence = batch[7].to(args.local_rank)
        B, T, C, H, W = lidar_sequence.shape
        
        x = lidar_sequence.to(torch.float32)
        x_folded = x.view(B * T, C, H, W)

        with torch.no_grad():
            loss = model(x_folded)

        current += B
        epoch_loss.append(loss.item())

        if dist.get_rank() == 0:
            logging.info(
                f"\rValid Progress: [{current:>6d}/{size*args.batch_size:>6d}] " +
                f"| Val MSE Loss: {np.mean(epoch_loss):>.6f} | " +
                f"{(time.time()-start_time)/current:>.4f}s/sample"
            )
            
    return epoch_loss


def main():
    log_path = f"/content/drive/MyDrive/PRCV/training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Initial Learning rate: {}".format(args.learning_rate))
    logging.info("Device rank: {}".format(args.local_rank))

    set_seed(args.seed)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # Initialize your Masked Autoencoder for 24-channel inputs
    model = HierarchicalLiDARCNNMAE(
        in_chans=args.in_chans,   # 24
        embed_dim=args.embed_dim, # 768
        mask_ratio=args.mask_ratio
    )

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Updated to ReduceLROnPlateau monitoring validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',         # Minimize reconstruction loss
        factor=0.5,         # Halve the learning rate
        patience=2,         # Wait 2 epochs without improvement before dropping
        threshold=1e-4, 
        verbose=True
    )
    
    curr_ep = 0
    if args.load_dir != '':
        model_path = log_path + args.load_dir
        model_ckpts = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_ckpts['model_states'])
        optimizer.load_state_dict(model_ckpts['optim_states'])
        curr_ep = model_ckpts['current_ep']
        # Note: ReduceLROnPlateau doesn't use standard scheduler.step(epoch) mapping
    
    # Datasets
    train_dataset = DrivingData(args.train_set+'/*')
    valid_dataset = DrivingData(args.valid_set+'/*')

    training_size = len(train_dataset)
    valid_size = len(valid_dataset)

    if dist.get_rank() == 0:
        logging.info(f'Length train: {training_size}; Valid: {valid_size}')

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    
    train_data = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=train_sampler, num_workers=args.workers, pin_memory=True
    )
    valid_data = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers, pin_memory=True
    )

    epochs = args.training_epochs

    for epoch in range(epochs):
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}")
        
        if epoch <= curr_ep and epoch != 0:
            continue

        train_data.sampler.set_epoch(epoch)
        valid_data.sampler.set_epoch(epoch)

        train_loss = training_epoch(train_data, model, optimizer, epoch, args)
        val_loss = validation_epoch(valid_data, model, epoch, args)

        mean_val_loss = np.mean(val_loss)

        log = {
            'epoch': epoch + 1, 
            'train_loss': np.mean(train_loss), 
            'val_loss': mean_val_loss,
            'lr': optimizer.param_groups[0]['lr']
        }

        if dist.get_rank() == 0:
            file_mode = 'w' if epoch == 0 else 'a'
            with open(log_path + 'train_log.csv', file_mode) as csv_file: 
                writer = csv.writer(csv_file) 
                if epoch == 0:
                    writer.writerow(log.keys())
                writer.writerow(log.values())
            
            save_state = {
                'optim_states': optimizer.state_dict(),
                'model_states': model.state_dict(),
                'current_ep': epoch
            }
            torch.save(save_state, log_path + f'epochs_{epoch}.pth')

        # Step the plateau scheduler using the validation loss
        scheduler.step(mean_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiDAR MAE Pretraining')
    parser.add_argument("--local_rank", type=int, default=0)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Keep small due to B*T folding")
    parser.add_argument("--training_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=3407)
    
    # Logging and paths
    parser.add_argument('--name', type=str, default="MAE_Pretraining_Run1")
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    
    # MAE Specific parameters
    parser.add_argument("--in_chans", type=int, default=24, help="Cropped Z-bins")
    parser.add_argument("--embed_dim", type=int, default=768, help="Bottleneck feature dimension")
    parser.add_argument("--patch_size", type=int, default=16, help="ViT spatial patch size")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Percentage of masked patches")
    
    args = parser.parse_args()
    main()