import sys
sys.path.append("..")
import glob
import torch
import argparse
import logging
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image

from model.GameFormer import GameFormer
from data_process import *
from utils.open_loop_test_utils import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils 

sys.argv = [a.replace('--local-rank', '--local_rank') for a in sys.argv]

class InteractionPredictionTestProcessor(DataProcess):
    def __init__(self, include_lidar=False):
        super().__init__()
        self.num_neighbors = 1
        self.hist_len = 11
        self.future_len = 80
        self.n_lanes = 6
        self.n_crosswalks = 4
        self.n_refline_waypoints = 1000
        self.include_lidar = include_lidar

    def extract_lidar_bev(self, points, x_min, x_max, y_min, y_max, z_min, z_max, xy_res, z_res):
        grid_x = round((x_max - x_min) / xy_res)
        grid_y = round((y_max - y_min) / xy_res)
        grid_z = round((z_max - z_min) / z_res)
        voxel_grid = np.zeros((grid_z, grid_x, grid_y), dtype=np.float32)

        x_vals = points[:, 0]
        y_vals = points[:, 1]
        z_vals = points[:, 2]

        x_bins = ((x_vals - x_min) / xy_res).astype(int)
        y_bins = ((y_vals - y_min) / xy_res).astype(int)
        z_bins = ((z_vals - z_min) / z_res).astype(int)

        mask = (
            (0 <= x_bins) & (x_bins < grid_x) &
            (0 <= y_bins) & (y_bins < grid_y) &
            (0 <= z_bins) & (z_bins < grid_z)
        )
        voxel_grid[z_bins[mask], x_bins[mask], y_bins[mask]] = 1.0
        return voxel_grid

    def normalize_lidar_points(self, pts, center, angle):
        pts_norm = pts.copy()
        pts_norm[:, 0] -= center[0]
        pts_norm[:, 1] -= center[1]
        
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        x_rot = cos_a * pts_norm[:, 0] - sin_a * pts_norm[:, 1]
        y_rot = sin_a * pts_norm[:, 0] + cos_a * pts_norm[:, 1]
        
        pts_norm[:, 0] = x_rot
        pts_norm[:, 1] = y_rot
        return pts_norm

    def get_lidar_point_at_timestep(self, scenario, curr_t):
        all_frames = []
        if len(scenario.compressed_frame_laser_data) == 0:
            return all_frames
            
        for i in range(curr_t - self.hist_len + 1, curr_t + 1):
            if i < 0 or i >= len(scenario.compressed_frame_laser_data):
                continue
                
            frame = scenario.compressed_frame_laser_data[i]
            if len(frame.lasers) == 0:
                continue
                
            pose = tf.constant(list(frame.pose.transform), dtype=tf.float64)
            pose = tf.reshape(pose, [4, 4])
            calibs = {c.name: c for c in frame.laser_calibrations}

            all_points = []
            for laser in frame.lasers:
                calib = calibs.get(laser.name)
                if calib is None: continue
                
                if laser.name == 1:
                    xyz_ri1, _, xyz_ri2, _ = womd_lidar_utils.extract_top_lidar_points(
                        laser, pose, calib)
                else:
                    xyz_ri1, _, xyz_ri2, _ = womd_lidar_utils.extract_side_lidar_points(
                        laser, calib)
                all_points.append(xyz_ri1.numpy())
                all_points.append(xyz_ri2.numpy())

            if len(all_points) > 0:
                pts = np.concatenate(all_points, axis=0)
                z_vehicle = pts[:, 2].copy()
                pose_np = pose.numpy()
                ones = np.ones((pts.shape[0], 1))
                pts_h = np.concatenate([pts, ones], axis=1)
                pts_global = (pose_np @ pts_h.T).T[:, :3]
                pts_global[:, 2] = z_vehicle
                all_frames.append(pts_global)
                
        return all_frames

    def process_frame(self, timestep, sdc_ids, tracks, scenario):      
        # 1. Fetch raw data for both interacting agents
        ego = self.ego_process(sdc_ids, tracks) 
        neighbors, neighbors_to_predict = self.neighbors_process(sdc_ids, tracks)
        
        # 2. Map Array Initialization matching GameFormer's expected (100, 16) dimensions
        agent_map_lanes = np.zeros(shape=(1+self.num_neighbors, self.n_lanes, 100, 16), dtype=np.float32)
        agent_map_crosswalk = np.zeros(shape=(1+self.num_neighbors, self.n_crosswalks, 100, 3), dtype=np.float32)

        # 3. Intercept and downsample maps for Primary Ego (Agent 1) and Interacting Agent (Agent 2)
        raw_lanes_0, agent_map_crosswalk[0] = self.map_process(ego[0])
        agent_map_lanes[0] = raw_lanes_0[:, ::3, :16] 
        
        raw_lanes_1, agent_map_crosswalk[1] = self.map_process(ego[1])
        agent_map_lanes[1] = raw_lanes_1[:, ::3, :16]
        
        # Maps for background neighbors
        for i in range(self.num_neighbors - 1):
            if neighbors[i, -1, 0] != 0:
                raw_lanes_n, agent_map_crosswalk[i+2] = self.map_process(neighbors[i])
                agent_map_lanes[i+2] = raw_lanes_n[:, ::3, :16]

        ground_truth = self.ground_truth_process(sdc_ids, tracks)
        
        # 4. Normalize all trajectories
        ego, neighbors, agent_map_lanes, agent_map_crosswalk, ground_truth, _ = self.normalize_data(
            ego, neighbors, agent_map_lanes, agent_map_crosswalk, ground_truth, viz=False)
            
        # 5. Repackage shapes to match GameFormer's expectations
        target_id = sdc_ids[1]
        target_idx = -1
        
        # Find where the target is hiding in the background neighbors
        for i, n_id in enumerate(neighbors_to_predict):
            if n_id == target_id:
                target_idx = i
                break
                
        # Swap it to index 0 safely (preserving relative coordinates!)
        if target_idx != -1 and target_idx != 0:
            neighbors[[0, target_idx]] = neighbors[[target_idx, 0]]
            neighbors_to_predict[0], neighbors_to_predict[target_idx] = neighbors_to_predict[target_idx], neighbors_to_predict[0]

        obs = {
            'ego_state': ego[0], 
            'neighbors_state': neighbors, 
            'map_lanes': agent_map_lanes, 
            'map_crosswalks': agent_map_crosswalk
        }

        # Conditional LiDAR Processing
        if self.include_lidar:
            center, angle = np.array(self.current_xyzh[0][:2]), self.current_xyzh[0][3]
            lidar_frames = self.get_lidar_point_at_timestep(scenario, timestep)
            
            if len(lidar_frames) > 0:
                bev_frames = []
                for lidar_pts in lidar_frames:
                    lidar_pts_norm = self.normalize_lidar_points(lidar_pts, center, angle)
                    bev = self.extract_lidar_bev(lidar_pts_norm, -75, 75, -75, 75, -2, 4, 0.5, 0.5)
                    bev_frames.append(bev)
                
                while len(bev_frames) < self.hist_len:
                    bev_frames.insert(0, np.zeros((12, 300, 300), dtype=np.uint8))
                    
                lidar_bev = np.array(bev_frames, dtype=np.uint8)
            else:
                lidar_bev = np.zeros((self.hist_len, 12, 300, 300), dtype=np.uint8)
            
            obs['lidar_bev'] = lidar_bev     
        
        return obs, neighbors_to_predict, ground_truth


def interaction_test():
    # logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')
    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use device: {}".format(args.device))
    logging.info("Include LiDAR: {}".format(args.include_lidar))

    # test files
    files = glob.glob(args.test_set+'/*')
    test_files = []
    test_id = 0
    
    # Ignore hidden files and folders like .ipynb_checkpoints
    required_files = [f for f in os.listdir(args.test_set) if not f.startswith('.')]
    
    for file in required_files:
        file_path = f"{args.test_set}/{file}"
        test_files.append(file_path)
        if file_path not in files:
            logging.error(f"File {file_path} does not exist!")
            sys.exit()

    processor = InteractionPredictionTestProcessor(include_lidar=args.include_lidar)

    scenario_ids = []
    collisions = []
    miss_rates = []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE = [], []

    check_point = torch.load(args.model_path, map_location="cpu")

    gameformer = GameFormer(
        modalities=args.modalities,
        encoder_layers=args.encoder_layers,
        decoder_levels=args.level,
        future_len=args.future_len,
        neighbors_to_predict=args.neighbors_to_predict
    )

    state_dict = check_point["model_states"]
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}

    gameformer.load_state_dict(state_dict, strict=True)
    gameformer = gameformer.to(args.device)
    gameformer.eval()

    for file in test_files:
        
        if not os.path.exists(file) or os.path.getsize(file) == 0:
            logging.warning(f"File {file} is empty or missing. Skipping.")
            continue
            
        try:
            scenarios = tf.data.TFRecordDataset(file)
        except Exception as e:
            logging.error(f"Failed to load TFRecord file {file}: {e}")
            continue

        valid_scenarios_in_file = 0

        for scenario in scenarios:
            parsed_data = scenario_pb2.Scenario()
            
            try:
                parsed_data.ParseFromString(scenario.numpy())
            except Exception as e:
                logging.error(f"Corrupted scenario encountered in {file}: {e}. Skipping.")
                continue
                
            scenario_id = parsed_data.scenario_id
            
            if len(parsed_data.tracks) == 0 or len(parsed_data.map_features) == 0:
                logging.warning(f"Scenario {scenario_id} is missing critical tracks or map features. Skipping.")
                continue

            if args.include_lidar and len(parsed_data.compressed_frame_laser_data) == 0:
                logging.warning(f"Scenario {scenario_id} is missing LiDAR data. Proceeding with zero-padding.")
            
            # Extract the two interacting tracks
            tracks_to_predict = [ids.track_index for ids in parsed_data.tracks_to_predict]
            
            if len(tracks_to_predict) < 2:
                logging.warning(f"Scenario {scenario_id} does not have 2 interacting agents. Skipping.")
                continue
                
            if max(tracks_to_predict) >= len(parsed_data.tracks):
                logging.error(f"Scenario {scenario_id}: tracks_to_predict index out of bounds. Skipping.")
                continue
                
            sdc_ids = [tracks_to_predict[0], tracks_to_predict[1]]
            
            valid_scenarios_in_file += 1
            test_id += 1
            logging.info(f"Testing scenario: {scenario_id}")
            timesteps = parsed_data.timestamps_seconds

            processor.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

            for curr_t in range(10, len(timesteps)-50, 5):
                logging.info(f"Testing timestep: {curr_t}")
                scenario_ids.append(f'{scenario_id}_{curr_t}')
                
                # Pass sdc_ids as a list
                data = processor.process_frame(curr_t, sdc_ids, parsed_data.tracks, parsed_data)
                
                if data is None:
                    continue
                else:
                    obs, neighbor_ids, gt_future = data

               
                # Force the visualizer to use the ID of the injected agent
                neighbor_ids = [sdc_ids[1]]
                
                inputs = {
                    'ego_state': torch.from_numpy(obs['ego_state']).unsqueeze(0).to(args.device),
                    'neighbors_state': torch.from_numpy(obs['neighbors_state']).unsqueeze(0).to(args.device),
                    'map_lanes': torch.from_numpy(obs['map_lanes']).unsqueeze(0).to(args.device),
                    'map_crosswalks': torch.from_numpy(obs['map_crosswalks']).unsqueeze(0).to(args.device)
                }

                if args.include_lidar:
                    # Permute from (Batch, Time, Z, H, W) -> (Batch, Z, Time, H, W)
                    inputs['lidar_bev'] = torch.from_numpy(obs['lidar_bev']).unsqueeze(0).permute(0, 2, 1, 3, 4).float().to(args.device)

                ego_future = gt_future[0]
                neighbors_future = gt_future[1:]

                with torch.no_grad():
                    level_k_outputs = gameformer(inputs)

                level = len(level_k_outputs.keys()) // 2 - 1
                trajectories = level_k_outputs[f'level_{level}_interactions']
                scores = level_k_outputs[f'level_{level}_scores']
                trajectories = select_future(trajectories, scores)
                plan = trajectories[0].cpu()
                predictions = trajectories[1:].cpu().numpy()
 
                current_state = inputs['ego_state'][0, -1].cpu()
                xy = torch.cat([current_state[None, :2], plan])
                dxy = torch.diff(xy, dim=0)
                theta = torch.atan2(dxy[:, 1], dxy[:, 0].clip(min=1e-3)).unsqueeze(-1)
                plan = torch.cat([plan, theta], dim=-1).numpy()
                collision = check_collision(plan, neighbors_future, obs['ego_state'][-1, 5:], 
                                            obs['neighbors_state'][:, -1, 5:])
                collisions.append(collision)
                miss = check_ego_miss(plan, ego_future)
                miss_rates.append(miss)
                logging.info(f"Ego Collision: {collision}, Miss: {miss}")

                similarity = check_ego_similarity(plan, ego_future)
                similarity_1s.append(similarity[9])
                similarity_3s.append(similarity[29])
                similarity_5s.append(similarity[49])
                logging.info(f"Ego Plan Similarity@1s: {similarity[9]}, Similarity@3s: {similarity[29]}, Similarity@5s: {similarity[49]}")

                prediction_error = check_agent_prediction(predictions, neighbors_future)
                prediction_ADE.append(prediction_error[0])
                prediction_FDE.append(prediction_error[1])
                logging.info(f"Prediction ADE: {prediction_error[0]}, FDE: {prediction_error[1]}")
                logging.info(f"--------------------------------------------------")

                if args.render:
                    # 1. Trim the padded zeros from the Predictions
                    ego_valid_len = np.sum(np.any(ego_future[:, :2] != 0, axis=-1))
                    neigh_valid_len = np.sum(np.any(neighbors_future[0][:, :2] != 0, axis=-1))
                    
                    trajectories_np = trajectories.cpu().numpy()
                    trimmed_trajs = [
                        trajectories_np[0][:max(1, ego_valid_len)],
                        trajectories_np[1][:max(1, neigh_valid_len)]
                    ]
                    
                    # 2. Trim the padded zeros from the Ground Truth
                    trimmed_gt = [
                        ego_future[:max(1, ego_valid_len), :2], 
                        neighbors_future[0][:max(1, neigh_valid_len), :2]
                    ]
                    
                    # 3. Extract Ego Pose (X, Y, Heading)
                    ego_xyh = [processor.current_xyzh[0][0], processor.current_xyzh[0][1], processor.current_xyzh[0][3]]
                    
                    # 4. Transform BOTH paths to global map coordinates
                    global_pred_trajectories = transform_to_global_frame(curr_t, trimmed_trajs, ego_xyh, neighbor_ids, parsed_data.tracks)
                    global_gt_trajectories = transform_to_global_frame(curr_t, trimmed_gt, ego_xyh, neighbor_ids, parsed_data.tracks)
                    
                    # 5. Package Map Features safely
                    map_features_dict = {
                        'lane': getattr(processor, 'lanes', {}),
                        'road_line': getattr(processor, 'roads', {}),
                        'road_edge': getattr(processor, 'road_edges', {}),
                        'crosswalk': getattr(processor, 'crosswalks', {}),
                        'speed_bump': getattr(processor, 'speed_bumps', {}),
                        'stop_sign': getattr(processor, 'stop_signs', {}),
                        'dynamic_map_states': parsed_data.dynamic_map_states
                    }

                    # 6. Actually draw and save the frame! (Passing the new global_gt_trajectories)
                   # Pass sdc_ids[0] so the visualizer knows exactly which car is the Ego!
                    plot_scenario(curr_t, sdc_ids[0], neighbor_ids, map_features_dict, ego_xyh, 
                                  parsed_data.tracks, global_pred_trajectories, global_gt_trajectories, args.name, scenario_id, args.save)

            # GIF Compilation Block
            if args.render and args.save:
                save_path = f"./testing_log/{args.name}/visualizations"
                frames = []
                png_files = sorted([f for f in os.listdir(save_path) if f.startswith(f"{scenario_id}_") and f.endswith('.png')],
                                   key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                for filename in png_files:
                    frames.append(Image.open(os.path.join(save_path, filename)))
                
                if frames:
                    gif_path = f"{save_path}/{scenario_id}_animation.gif"
                    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0)
                    logging.info(f"Successfully saved scenario GIF: {gif_path}")

            
            
        if valid_scenarios_in_file == 0:
            logging.warning(f"File {file} processed, but yielded 0 valid interacting scenarios.")

    df = pd.DataFrame(data={'scenarios': scenario_ids, 'collision': collisions, 'miss': miss_rates, 
                            'Prediction_ADE': prediction_ADE, 'Prediction_FDE': prediction_FDE,
                            'Human_L2_1s': similarity_1s, 'Human_L2_3s': similarity_3s, 'Human_L2_5s': similarity_5s})
    df.to_csv(f'./testing_log/{args.name}/testing_log.csv')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Interaction Prediction Testing')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--level", type=int, help='decoder reasoning levels (K)', default=3)
    parser.add_argument("--neighbors_to_predict", type=int, help='neighbors to predict, 1 for Waymo Joint Prediction', default=1)
    parser.add_argument("--modalities", type=int, help='joint num of modalities', default=6)
    parser.add_argument("--future_len", type=int, help='prediction horizons', default=80)
    parser.add_argument("--encoder_layers", type=int, help='encoder layers', default=6)
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--test_set', type=str, help='path to testing datasets')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--render', action="store_true", help='if render the scenario', default=False)
    parser.add_argument('--save', action="store_true", help='if save the rendered images', default=False)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    parser.add_argument('--eval_scenario_id', type=int, help='scenario ID to evaluate', default=None)
    parser.add_argument('--include_lidar', action="store_true", help='whether to include LiDAR BEV data in the model inputs', default=True) 
    args = parser.parse_args()

    interaction_test()