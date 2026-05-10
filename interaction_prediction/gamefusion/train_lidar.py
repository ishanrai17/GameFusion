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

from utils.inter_pred_utils import *
from interaction_prediction.gamefusion.utilities import DrivingData

# Import your Hierarchical CNN MAE (SparK style)
from model.modules import HierarchicalLiDARCNNMAE

scaler = torch.cuda.amp.GradScaler()
def training_epoch(train_data, model, optimizer, epoch, args):
    epoch_loss = []
    model.train()
    total_samples = 0
    start_time = time.time()
    
    # Define how many distinct scenes to accumulate before stepping
    # Accumulating over 4 steps means updating weights based on 4 completely different map locations
    accumulation_steps = 4  

    for idx, batch in enumerate(train_data):
        lidar_sequence = batch.to(args.local_rank)
        B, C, T, H, W = lidar_sequence.shape
        
        x = lidar_sequence.to(torch.float32)
        x_permuted = x.permute(0, 2, 1, 3, 4).contiguous()
        x_folded = x_permuted.view(B * T, C, H, W)

        # We do NOT zero the gradients here anymore!
        # optimizer.zero_grad()  <-- REMOVED

        # Forward pass
        loss = model(x_folded)
        
        # Normalize the loss mathematically to account for the accumulation summation
        normalized_loss = loss / accumulation_steps
        
        # Accumulate gradients into the autograd buffers
        normalized_loss.backward()
        
        # Only update weights once we have accumulated enough diverse scenes
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_data):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad() # Flush buffers strictly AFTER stepping

        total_samples += B
        epoch_loss.append(loss.item()) # Keep logging the raw un-normalized loss

    if dist.get_rank() == 0:
        elapsed_time = time.time() - start_time
        logging.info(
            f"Train Epoch {epoch+1} Summary | MSE Loss: {np.mean(epoch_loss):.6f} | " +
            f"Total Time: {elapsed_time:.2f}s ({elapsed_time/total_samples:.4f}s/sample)"
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
        
        # Ensure validation perfectly mirrors the corrected training dimensions
        B, C, T, H, W = lidar_sequence.shape
        
        x = lidar_sequence.to(torch.float32)
        x_permuted = x.permute(0, 2, 1, 3, 4).contiguous()
        x_folded = x_permuted.view(B * T, C, H, W)

        with torch.no_grad():
            loss = model(x_folded)

        current += B
        epoch_loss.append(loss.item())

        # CRITICAL FIX: Log exactly ONCE here, after the entire validation loop finishes
        if dist.get_rank() == 0:
            elapsed_time = time.time() - start_time
            logging.info(
                f"Valid Epoch {epoch+1} Summary | Val MSE Loss: {np.mean(epoch_loss):.6f} | " +
                f"Total Time: {elapsed_time:.2f}s"
            )
        del lidar_sequence, x, x_permuted, x_folded, loss
        torch.cuda.empty_cache()
            
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

    # Instantiate your Hierarchical CNN MAE
    model = HierarchicalLiDARCNNMAE(
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio
    )

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',         
        factor=0.5,         
        patience=2,         
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
    
    train_dataset = DrivingData(args.train_set+'/*')
    valid_dataset = DrivingData(args.valid_set+'/*')

    training_size = len(train_dataset)
    valid_size = len(valid_dataset)

    if dist.get_rank() == 0:
        logging.info(f'Length train: {training_size}; Valid: {valid_size}')

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    
    # Kept pin_memory=True to maximize GPU transfer speeds for the uint8 bytes
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

        scheduler.step(mean_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiDAR CNN-MAE Pretraining')
    parser.add_argument("--local_rank", type=int, default=0)
    
    # Set conservative defaults to prevent host RAM out-of-memory errors
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--training_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=3407)
    
    parser.add_argument('--name', type=str, default="CNN_MAE_Pretraining_Run1")
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)
    parser.add_argument("--workers", type=int, default=1)
    
    parser.add_argument("--in_chans", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    
    args = parser.parse_args()
    main()