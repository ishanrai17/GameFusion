"""
  Rohith Kumar Senthil Kumar
  Ishan Rai
  Ying-Jen Chiang
  5330 Computer Vision
  Final Project
    Open Loop Testing Utilities for GameFormer: A Multi-Level Transformer for Interaction Prediction in Autonomous Driving
"""

import os
import logging
import torch
import numpy as np
import scipy.spatial as T
import matplotlib as mpl
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate


def initLogging(log_file: str, level: str = "INFO"):
    """Initialize logging configuration to log messages to both a file and the console with a consistent format."""
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def select_future(trajectories, scores):
    """Select the future trajectory for each agent based on the highest predicted score."""
    trajectories = trajectories.squeeze(0)
    scores = scores.squeeze(0)
    best_mode = torch.argmax(scores, dim=-1)
    best_mode_future = trajectories[torch.arange(best_mode.shape[0])[:, None], best_mode.unsqueeze(-1), :, :2]
    best_mode_future = best_mode_future.squeeze(1)

    return best_mode_future


def transform_to_global_frame(timestep, trajectories, ego_pose, predict_ids, tracks):
    """Transform trajectories to the global frame."""
    ego_p = ego_pose[:2]
    ego_h = ego_pose[2]

    global_trajectories = []   
    line = LineString(trajectories[0])
    line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
    line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
    line = np.array(line.coords)
    traj = np.insert(line, 0, ego_p, axis=0)
    traj = trajectory_smoothing(traj)
    global_trajectories.append(traj)

    for i, j in enumerate(predict_ids):
        current_state = np.array([tracks[j].states[timestep].center_x, tracks[j].states[timestep].center_y])
        line = LineString(trajectories[i+1])
        line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
        line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
        line = np.array(line.coords)
        traj = np.insert(line, 0, current_state, axis=0)
        traj = trajectory_smoothing(traj)
        global_trajectories.append(traj)

    return global_trajectories


def transform_to_global_frame_multi_modal(trajectories, ego_pose):
    """Transform trajectories to the global frame for multiple agents and multiple modes."""
    ego_p = ego_pose[0][:2]
    ego_h = ego_pose[1]

    global_trajectories = []
    for i in range(trajectories.shape[0]):
        multi_trajectories = []
        for j in range(trajectories.shape[1]):
            line = LineString(trajectories[i, j, :, :2].copy())
            line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
            line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
            line = np.array(line.coords)
            line = trajectory_smoothing(line)
            multi_trajectories.append(line)

        global_trajectories.append(multi_trajectories)

    return global_trajectories


def trajectory_smoothing(trajectory):
    """Apply smoothing to the trajectory."""
    x = trajectory[:,0]
    y = trajectory[:,1]

    window_length = 25
    
    # Safety check to prevent errors if the trajectory is shorter than the window length
    if len(trajectory) < window_length:
        window_length = len(trajectory) if len(trajectory) % 2 != 0 else len(trajectory) - 1
        
    if window_length > 3:
        x = signal.savgol_filter(x, window_length=window_length, polyorder=3)
        y = signal.savgol_filter(y, window_length=window_length, polyorder=3)
   
    return np.column_stack([x, y])

def _plot_ground_truth(gt_trajectories):
    """Plot the ground truth trajectories for the agents in the scenario."""
    for i, traj in enumerate(gt_trajectories):
        # Draw bold, bright green lines for ground truth so it pops against the white lanes
        plt.plot(traj[:, 0], traj[:, 1], color='#22c55e', lw=3.5, alpha=1.0, solid_capstyle='round', zorder=4)

# Add 'gt_trajectories' right after 'trajectories'
def plot_scenario(timestep, sdc_id, predict_ids, map_features, ego_pose, agents, trajectories, gt_trajectories, name, scenario_id, save=False):
    """Visualize the scenario with map features, ground truth trajectories, predicted trajectories, and agent positions. The visualization includes a custom legend and timestep display for better interpretability."""
    plt.ion()
    fig = plt.gcf()
    dpi = 100
    size_inches = 800 / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_tight_layout(True)

    _plot_map_features(map_features)
    _plot_traffic_signals(map_features['dynamic_map_states'][timestep])
    
    _plot_ground_truth(gt_trajectories)
    _plot_trajectories(trajectories)
    _plot_agents(agents, timestep, sdc_id, predict_ids)

    # ─── NEW: LEGEND AND TIMESTEP LABELS ─────────────────────────────────────
    
    # 1. Timestep Display (Top Right, semi-transparent black box)
    plt.text(0.97, 0.97, f"Timestep: {timestep}", 
             transform=plt.gca().transAxes, fontsize=14, fontweight='bold', color='white', 
             ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=5), zorder=10)

    # 2. Custom Legend (Top Left, matching dark mode aesthetic)
    # 2. Custom Legend (Top Left, matching dark mode aesthetic)
    legend_elements = [
        # Changed markerfacecolor to bright red!
        Line2D([0], [0], marker='o', color='w', label='Ego Prediction', markerfacecolor='#ef4444', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', label='Target Prediction', markerfacecolor='#eab308', markersize=9, linestyle='None'),
        Line2D([0], [0], color='#22c55e', lw=3.5, label='Ground Truth (Actual)')
    ]
    
    plt.gca().legend(handles=legend_elements, loc='upper left', framealpha=0.6, facecolor='#0f172a', 
                     edgecolor='none', labelcolor='white', fontsize=12, prop={'weight':'bold'})
                     
    # ────────────────────────────────────────────────────────────────────────

   
    plt.gca().set_facecolor('#64748b')
    plt.gca().margins(0)  
    plt.gca().set_aspect('equal')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axis([-60 + ego_pose[0], 60 + ego_pose[0], -60 + ego_pose[1], 60 + ego_pose[1]])

    if save:
        save_path = f"./testing_log/{name}/visualizations"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{scenario_id}_{timestep}.png', bbox_inches='tight')
    else:
        plt.pause(1)

    plt.clf()


def _plot_agents(tracks, timestep, sdc_id, predict_ids):
    """Plot the agents in the scenario, differentiating between the ego vehicle, target vehicles, and background neighbors using distinct colors and z-ordering to ensure clarity in the visualization."""
    for id, track in enumerate(tracks):
        if not track.states[timestep].valid:
            continue
        else:
            state = track.states[timestep]
            pos_x, pos_y = state.center_x, state.center_y
            length, width = state.length, state.width

            """if id in predict_ids:
                color = '#eab308' # NEIGHBOR: Waymax Yellow/Orange
                zorder = 4
            elif id == sdc_id:
                color = '#06b6d4' # EGO: Waymax Cyan
                zorder = 4
            else:
                color = '#64748b' # BACKGROUND AGENTS: Muted Gray
                zorder = 3"""
				
            # Determine vehicle color based on its ID
            if id == sdc_id:
              color = 'white'      # Ego Vehicle
              zorder = 10         # Draw Ego on top
            elif id in predict_ids:
              color = '#eab308'    # Target Vehicle (Yellow to match legend)
              zorder = 9          # Draw Target just below Ego
            else:
              color = 'black'      # Background Neighbors
              zorder = 8          # Draw background cars lowest

            rect = plt.Rectangle((pos_x - length/2, pos_y - width/2), length, width, linewidth=2, color=color, alpha=0.9, zorder=zorder,
                                 transform=mpl.transforms.Affine2D().rotate_around(*(pos_x, pos_y), state.heading) + plt.gca().transData)
            plt.gca().add_patch(rect)


def _plot_trajectories(trajectories):
    """Plot the predicted trajectories for the agents, using distinct colors to differentiate between the ego vehicle's prediction and the target vehicles' predictions, ensuring that the visualization is clear and matches the legend for better interpretability."""
    for i, traj in enumerate(trajectories):
        if i == 0:
            # Ego Prediction -> Red to match the legend
            plt.scatter(traj[:, 0], traj[:, 1], c='#ef4444', s=15, zorder=5)
        else:
            # Target Prediction -> Yellow to match the legend
            plt.scatter(traj[:, 0], traj[:, 1], c='#eab308', s=15, zorder=5)


def _plot_map_features(map_features):
    """Plot the map features such as lanes, road lines, road edges, crosswalks, speed bumps, and stop signs with distinct styles and colors to enhance the visualization of the scenario's environment."""
    for lane in map_features.get("lane", {}).values():
        pts = np.array([[p.x, p.y] for p in lane.polyline])
        plt.plot(pts[:, 0], pts[:, 1], linestyle=":", color="gray", linewidth=2)

    for road_line in map_features.get("road_line", {}).values():
        pts = np.array([[p.x, p.y] for p in road_line.polyline])
        if road_line.type == 1:
            plt.plot(pts[:, 0], pts[:, 1], 'w', linestyle='dashed', linewidth=2)
        elif road_line.type == 2:
            plt.plot(pts[:, 0], pts[:, 1], 'w', linestyle='solid', linewidth=2)
        elif road_line.type == 3:
            plt.plot(pts[:, 0], pts[:, 1], 'w', linestyle='solid', linewidth=2)
        elif road_line.type == 4:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=2)
        elif road_line.type == 5:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=2)
        elif road_line.type == 6:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=2)
        elif road_line.type == 7:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=2)
        elif road_line.type == 8:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='dotted', linewidth=2)
        else:
            plt.plot(pts[:, 0], pts[:, 1], 'k', linewidth=2)

    for road_edge in map_features.get("road_edge", {}).values():
        pts = np.array([[p.x, p.y] for p in road_edge.polyline])
        plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=2)

    for crosswalk in map_features.get("crosswalk", {}).values():
        poly_points = [[p.x, p.y] for p in crosswalk.polygon]
        poly_points.append(poly_points[0])
        pts = np.array(poly_points)
        plt.plot(pts[:, 0], pts[:, 1], 'b:', linewidth=2)

    for speed_bump in map_features.get("speed_bump", {}).values():
        poly_points = [[p.x, p.y] for p in speed_bump.polygon]
        poly_points.append(poly_points[0])
        pts = np.array(poly_points)
        plt.plot(pts[:, 0], pts[:, 1], 'xkcd:orange', linewidth=2)

    for stop_sign in map_features.get("stop_sign", {}).values():
        plt.scatter(stop_sign.position.x, stop_sign.position.y, marker="8", s=100, c="red")


def _plot_traffic_signals(dynamic_map_features):
    """Plot the traffic signals (traffic lights) on the map, using colored circles to represent the state of each traffic light (red, yellow, green) based on the dynamic map features provided for the current timestep."""
    if not hasattr(dynamic_map_features, 'lane_states'):
        return
        
    for lane_state in dynamic_map_features.lane_states:
        stop_point = lane_state.stop_point

        if lane_state.state in [1, 4, 7]:
            state = 'r' 
        elif lane_state.state in [2, 5, 8]:
            state = 'y'
        elif lane_state.state in [3, 6]:
            state = 'g'
        else:
            state = None

        if state:
            light = plt.Circle((stop_point.x, stop_point.y), 1.2, color=state)
            plt.gca().add_patch(light)


def wrap_to_pi(theta):
    """Wrap an angle to the range [-pi, pi)."""
    return (theta+np.pi) % (2*np.pi) - np.pi


def return_circle_list(x, y, l, w, yaw):
    """Return a list of circle centers representing the vehicle's shape for collision checking, based on the vehicle's position, dimensions, and orientation."""
    r = w/np.sqrt(2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    if l < 4.0:
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c1, c2]
    elif l >= 4.0 and l < 8.0:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c0, c1, c2]
    else:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c3 = [x-(l-w)/2*cos_yaw/2, y-(l-w)/2*sin_yaw/2]
        c4 = [x+(l-w)/2*cos_yaw/2, y+(l-w)/2*sin_yaw/2]
        c = [c0, c1, c2, c3, c4]

    for i in range(len(c)):
        c[i] = np.stack(c[i], axis=-1)

    c = np.stack(c, axis=-2)

    return c


def return_collision_threshold(w1, w2):
    """Return the collision threshold based on the widths of two vehicles."""
    return (w1 + w2) / np.sqrt(3.8)


def check_collision(ego_center_points, neighbor_center_points, ego_size, neighbors_size):
    """Check for collisions between the ego vehicle and neighboring vehicles over the predicted trajectory, using the circle-based representation of the vehicles and a distance threshold to determine if a collision occurs at any timestep."""
    collision = False

    for t in range(ego_center_points.shape[0]):
        if check_collision_step(ego_center_points[t], neighbor_center_points[:, t], ego_size, neighbors_size):
            collision = True

    return collision 


def check_collision_step(ego_center_points, neighbor_center_points, ego_size, neighbors_size):
    """Check for collision at a single timestep between the ego vehicle and neighboring vehicles, using the circle-based representation of the vehicles and a distance threshold to determine if a collision occurs."""
    collision = []
    plan_x, plan_y, plan_yaw = ego_center_points[0], ego_center_points[1], ego_center_points[2], 
    plan_l, plan_w = ego_size[0], ego_size[1]
    ego_vehicle = return_circle_list(plan_x, plan_y, plan_l, plan_w, plan_yaw)  

    for i in range(neighbor_center_points.shape[0]):
        neighbor_length = neighbors_size[i, 0]
        neighbor_width = neighbors_size[i, 1]

        if neighbor_center_points[i, 0] != 0:
            neighbor_vehicle = return_circle_list(neighbor_center_points[i, 0], neighbor_center_points[i, 1], 
                                                  neighbor_length, neighbor_width, neighbor_center_points[i, 2])
            distance = [np.linalg.norm(ego_vehicle[i] - neighbor_vehicle[j], axis=-1) 
                        for i in range(ego_vehicle.shape[0]) for j in range(neighbor_vehicle.shape[0])]
            distance = np.stack(distance, axis=-1)
            threshold = return_collision_threshold(plan_w, neighbor_width)
            collision.append(np.any(distance < threshold))

    return np.any(collision)


def check_ego_miss(traj, route):
    """Check if the ego vehicle deviates from the route."""
    distance_to_ref = T.distance.cdist(traj[:, :2], route[:, :2])
    distance_to_route = np.min(distance_to_ref, axis=-1)

    if np.any(distance_to_route > 4.5):
        off_route = True
    else:
        off_route = False

    return off_route


def check_ego_similarity(traj, gt):
    """Check the similarity between the ego vehicle's trajectory and the ground truth."""
    error = np.linalg.norm(traj[:, :2] - gt[:, :2], axis=-1)
    
    return error


def check_agent_prediction(trajs, gt):
    """Check the accuracy of the predicted trajectories for the agents by calculating the Average Displacement Error (ADE) and Final Displacement Error (FDE) compared to the ground truth trajectories, while handling cases where some agents may not be present in the ground truth."""
    ADE = []
    FDE = []
    mask = np.not_equal(gt[:, :, :2], 0)

    # Dynamically loop over the number of agents provided
    for i in range(gt.shape[0]):
        if mask[i, 0, 0]:
            error = np.linalg.norm(trajs[i, :, :2] - gt[i, :, :2], axis=-1) 
            error = error * mask[i, :, 0]
            ADE.append(np.mean(error))
            FDE.append(error[-1])

    # Safety catch
    if len(ADE) == 0:
        return 0.0, 0.0

    return np.mean(ADE), np.mean(FDE)


def inverse_dynamics(traj, curr_state):
    """Calculate the control inputs (acceleration and steering) required to follow a given trajectory from the current state, using a simple bicycle model for the vehicle dynamics and ensuring that the control inputs are within realistic limits for acceleration and steering."""
    dt = 0.1
    max_delta = 0.5 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    traj = torch.cat([curr_state[:, :2], traj], dim=0)
    d_xy = torch.diff(traj, dim=0)
    v = torch.norm(d_xy, dim=-1) / dt
    theta = torch.arctan2(d_xy[:, 1], d_xy[:, 0].clamp(min=1e-3))
    a = torch.diff(v, dim=-1) / dt
    delta = torch.diff(theta, dim=-1) / dt
    a = torch.cat([a, a[-1, None]]).clamp(-max_a, max_a)
    delta = torch.cat([delta, delta[-1, None]]).clamp(-max_delta, max_delta)
    act = torch.stack([a, delta], dim=-1)

    return act


def bicycle_model(control, current_state):
    """Simulate the bicycle model for vehicle dynamics."""
    dt = 0.1 # discrete time period [s]
    max_delta = 0.6 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_delta, max_delta) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * delta
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj