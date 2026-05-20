import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import sys

# Setup paths (Adjust if running outside your standard directory)
sys.path.insert(0, '/content/GameFusion')
sys.path.insert(1, '/content/GameFusion/interaction_prediction')

# Import your custom Model
from model.modules import HierarchicalLiDARCNNMAE

def mine_hard_examples(
    model_path, 
    unlabeled_folder, 
    output_npz_path, 
    top_k_percent=0.25  # Mine the top 25% hardest files
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from: {model_path}")

    # 1. Initialize Model and Load Weights
    model = HierarchicalLiDARCNNMAE(in_chans=24, embed_dim=768, mask_ratio=0.75)
    
    # Load DDP checkpoint safely into a standard model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle 'module.' prefix if the model was saved inside DDP
    state_dict = checkpoint['model_states']
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval()

    # 2. Locate all unlabeled candidate files
    candidate_files = glob.glob(os.path.join(unlabeled_folder, '*.npz'))
    print(f"Found {len(candidate_files)} candidate scenarios in {unlabeled_folder}.")

    scored_scenarios = []

    # 3. Offline Inference Loop
    with torch.no_grad():
        for file_path in tqdm(candidate_files, desc="Scoring Scenarios"):
            try:
                # Load the data and gracefully close the handle
                with np.load(file_path, allow_pickle=True) as data:
                    # Extract the raw array and transpose to [C, T, H, W]
                    lidar_bev = data['lidar_bev']
                    lidar_bev = np.transpose(lidar_bev, (1, 0, 2, 3))
                    
                    # Extract scenario_id (Assuming it is saved inside the npz)
                    # If it's just the filename, use: os.path.basename(file_path).split('.')[0]
                    scenario_id = str(data['scenario_id']) 
                
                # Convert to tensor and add the Batch dimension: [1, 24, 11, 512, 512]
                lidar_sequence = torch.from_numpy(lidar_bev).unsqueeze(0).to(device)
                
                # Apply the exact same folding logic from your train.py
                B, C, T, H, W = lidar_sequence.shape
                x = lidar_sequence.to(torch.float32)
                x_permuted = x.permute(0, 2, 1, 3, 4).contiguous()
                x_folded = x_permuted.view(B * T, C, H, W)
                
                # Protect your 15GB VRAM with AMP
                with torch.cuda.amp.autocast():
                    # Request the explicit decomposition to get the Foreground MSE
                    loss, fg_loss, bg_loss = model(x_folded, return_decomposition=True)
                
                # Record the Foreground MSE as the "Struggle Score"
                scored_scenarios.append({
                    'scenario_id': scenario_id,
                    'file_path': file_path,
                    'fg_mse': fg_loss.item()
                })
                
                # Explicit cleanup to prevent iteration memory leaks
                del lidar_sequence, x, x_permuted, x_folded, loss, fg_loss, bg_loss
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    # 4. Rank and Filter the Hardest Examples
    # Sort descending so the highest Foreground MSE (hardest) is at the top
    scored_scenarios.sort(key=lambda x: x['fg_mse'], reverse=True)
    
    num_to_keep = int(len(scored_scenarios) * top_k_percent)
    hard_scenarios = scored_scenarios[:num_to_keep]
    
    print(f"\n--- Active Learning Mining Complete ---")
    print(f"Top hardest score (FG MSE): {hard_scenarios[0]['fg_mse']:.4f}")
    print(f"Cutoff score (FG MSE): {hard_scenarios[-1]['fg_mse']:.4f}")
    
    # 5. Extract IDs and Paths for exporting
    mined_ids = np.array([item['scenario_id'] for item in hard_scenarios])
    mined_paths = np.array([item['file_path'] for item in hard_scenarios])
    
    # Save to a new .npz file that your dataloader can consume
    np.savez(
        output_npz_path, 
        scenario_ids=mined_ids, 
        file_paths=mined_paths
    )
    
    print(f"Successfully saved {len(mined_ids)} hard-mined scenarios to: {output_npz_path}")

if __name__ == "__main__":
    # Define your paths
    MODEL_CHECKPOINT = "/content/drive/MyDrive/PRCV/training_log/CNN_MAE_Pretraining_Run1/epochs_0.pth"
    UNLABELED_SHARD_DIR = "/content/drive/MyDrive/PRCV/data/next_shard_folder"
    OUTPUT_FILE = "/content/drive/MyDrive/PRCV/data/hard_mined_scenarios_shard2.npz"
    
    mine_hard_examples(
        model_path=MODEL_CHECKPOINT,
        unlabeled_folder=UNLABELED_SHARD_DIR,
        output_npz_path=OUTPUT_FILE,
        top_k_percent=0.25  # Adjust this depending on your GPU time budget
    )