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
        self.num_neighbors = 10
        self.hist_len = 11
        self.future_len = 50
        self.n_lanes = 6
        self.n_crosswalks = 4
        self.n_refline_waypoints = 1000
        self.include_lidar = include_lidar # Added flag to class

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
            
        # Get history frames: curr_t - 10 to curr_t
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

    def process_frame(self, timestep, sdc_id, tracks, scenario):      
        ego = self.ego_process([sdc_id], tracks) 
        neighbors, neighbors_to_predict = self.neighbors_process([sdc_id], tracks)
        agent_map_lanes = np.zeros(shape=(1+self.num_neighbors, self.n_lanes, 300, 17), dtype=np.float32)
        agent_map_crosswalk = np.zeros(shape=(1+self.num_neighbors, self.n_crosswalks, 100, 3), dtype=np.float32)

        agent_map_lanes[0], agent_map_crosswalk[0] = self.map_process(ego[0])
        for i in range(self.num_neighbors):
            if neighbors[i, -1, 0] != 0:
                agent_map_lanes[i+1], agent_map_crosswalk[i+1] = self.map_process(neighbors[i])

        ground_truth = self.ground_truth_process([sdc_id], tracks)
        
        ego, neighbors, map_lanes, map_crosswalks, ground_truth = self.normalize_data(
            ego, neighbors, agent_map_lanes, agent_map_crosswalk, ground_truth, viz=False)
            
        obs = {
            'ego_state': ego, 
            'neighbors_state': neighbors, 
            'map_lanes': map_lanes, 
            'map_crosswalks': map_crosswalks
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
                
                # Pad if missing frames at the start of the scenario
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
    required_files = os.listdir(args.test_set)
    for file in required_files:
        file_path = f"{args.test_set}/{file}"
        test_files.append(file_path)
        if file_path not in files:
            logging.error(f"File {file_path} does not exist!")
            sys.exit()

    # data processor with conditional flag
    processor = InteractionPredictionTestProcessor(include_lidar=args.include_lidar)

    # cache results
    scenario_ids = []
    collisions = []
    miss_rates = []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE = [], []

    # load model
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

    # iterate thru test files
    for file in test_files:
        scenarios = tf.data.TFRecordDataset(file)

        # iterate thru scenarios
        for scenario in scenarios:
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())
            scenario_id = parsed_data.scenario_id
            
            test_id += 1
            logging.info(f"Testing scenario: {scenario_id}")
            sdc_id = parsed_data.sdc_track_index
            timesteps = parsed_data.timestamps_seconds

            # build map
            processor.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

            # iterate thru timesteps
            for curr_t in range(10, len(timesteps)-50, 5):
                logging.info(f"Testing timestep: {curr_t}")
                scenario_ids.append(f'{scenario_id}_{curr_t}')
                
                # Passed parsed_data (scenario) to process_frame to extract LiDAR if flag is True
                data = processor.process_frame(curr_t, sdc_id, parsed_data.tracks, parsed_data)
                if data is None:
                    continue
                else:
                    obs, neighbor_ids, gt_future = data

                # prepare data base dictionary
                inputs = {
                    'ego_state': torch.from_numpy(obs['ego_state']).unsqueeze(0).to(args.device),
                    'neighbors_state': torch.from_numpy(obs['neighbors_state']).unsqueeze(0).to(args.device),
                    'map_lanes': torch.from_numpy(obs['map_lanes']).unsqueeze(0).to(args.device),
                    'map_crosswalks': torch.from_numpy(obs['map_crosswalks']).unsqueeze(0).to(args.device)
                }

                # Conditionally append LiDAR to inputs
                if args.include_lidar:
                    inputs['lidar_bev'] = torch.from_numpy(obs['lidar_bev']).unsqueeze(0).float().to(args.device)

                ego_future = gt_future[0]
                neighbors_future = gt_future[1:]

                # level-k reasoning
                with torch.no_grad():
                    level_k_outputs = gameformer(inputs)

                level = len(level_k_outputs.keys()) // 2 - 1
                trajectories = level_k_outputs[f'level_{level}_interactions']
                scores = level_k_outputs[f'level_{level}_scores']
                trajectories = select_future(trajectories, scores)
                plan = trajectories[0].cpu()
                predictions = trajectories[1:].cpu().numpy()
 
                # compute metrics
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

                # plot scenario
                if args.render:
                    trajectories = transform_to_global_frame(curr_t, trajectories.cpu().numpy(), 
                                                             processor.current_xyh, neighbor_ids, parsed_data.tracks)
                    plot_scenario(curr_t, sdc_id, neighbor_ids, processor.map, processor.current_xyh, 
                                  parsed_data.tracks, trajectories, args.name, scenario_id, args.save)
            break
            
    # save results
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
    parser.add_argument('--include_lidar', action="store_true", help='whether to include LiDAR BEV data in the model inputs', default=True) # The new flag
    args = parser.parse_args()

    interaction_test()