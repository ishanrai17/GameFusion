# Interaction Prediction with GameFusion: Multi-Camera Visual Token Fusion

**Authors:** Ishan Rai, Ying-Jen Chiang, Rohith Kumar Senthil Kumar
**Emails:** rai.ish@northeastern.edu, chiang.yin@northeastern.edu, senthilkumar.ro@northeastern.edu
**Course:** CS 5330 — Computer Vision (Spring 2026)
**Date:** April 23, 2026

**IDE used:** Visual Studio Code
**Operating System:** macOS

**Video Presentation Link:** https://drive.google.com/file/d/119hFKyXesuK86i0DPosKnHm0ubqnZunR/view?usp=sharing


# GameFusion

A fork of [GameFormer](https://github.com/MCZhi/GameFormer) that adds **multi-camera visual token fusion** to the game-theoretic interaction prediction pipeline. Built on the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/).

The original GameFormer predicts interacting agents using only trajectory history and HD map context. Our project, "Interaction Prediction with GameFormer", focuses on enhancing the GameFormer model by incorporating camera and LiDAR tokens as additional inputs. The goal is to evaluate the impact of these modalities on the model's performance in predicting interactions in autonomous driving scenarios, aiming to improve accuracy in complex environments.

---

## What We Added: Camera Token Fusion

The core contribution is an end-to-end pipeline that ingests Waymo's discretized camera tokens and fuses them into the existing GameFormer architecture. Every component below is new.

### Camera Token Data Pipeline

Waymo provides pre-computed discretized visual tokens for each camera frame — a compact integer-sequence representation of what each camera sees. Our data processing extracts these from the raw TFRecords and packages them for training.

**Extraction** (`interaction_prediction/data_process.py`, `extract_camera_tokens`):

1. For each scenario, check if `frame_camera_tokens` are embedded in the protobuf. If not, fall back to loading a per-scenario TFRecord from a separate `--camera_dir`.
2. Extract tokens for the **11 history frames** across **8 cameras**, each producing up to **256 discrete tokens**.
3. Apply a **+1 offset** to all token IDs — this reserves ID 0 as a padding index for the embedding layer, so the model can distinguish "no token" from "token 0."
4. Store the result as a `(11, 8, 256)` int32 array in each `.npz` file.

**Graceful fallback**: If camera tokens are unavailable for a scenario (missing file, empty protobuf), the array is all-zeros. The model handles this transparently via padding masks — it simply operates without camera context for that sample.

The data flows from raw Waymo TFRecords through `extract_camera_tokens`, which parses `frame_camera_tokens` from the protobuf (or loads from a separate per-scenario TFRecord as fallback). The resulting `(11, 8, 256)` int32 array with +1 offset is saved into the `.npz` file, loaded by `DrivingData.__getitem__()` as the 8th element of each batch, and passed to the model as `inputs['camera_tokens']`.

### CameraTokenEncoder Architecture

The encoder (`model/modules.py`, `CameraTokenEncoder`) converts the raw token IDs into a fixed-size representation per camera through a four-stage pipeline. The input is a `(B, 11, 8, 256)` tensor of integer token IDs.

**Stage 1 — Embedding:** Each token ID is embedded via a learned vocabulary embedding (8193 entries, 128-dim, padding_idx=0). Temporal embeddings (which of the 11 frames) and camera embeddings (which of the 8 views) are added, then an MLP projects the combined representation to 256-dim. Output shape: `(B, 11, 8, 256, 256)`.

**Stage 2 — Spatial Pooling:** A learned gated softmax aggregates across the 256 tokens per camera per timestep. Padding tokens are masked to `-1e9` before the softmax so they contribute zero weight. This collapses the token dimension, producing `(B, 11, 8, 256)`.

**Stage 3 — Cross-Camera Self-Attention:** A single TransformerEncoderLayer (4 heads, 256-dim, GELU, dropout 0.1) is applied per-timestep across the 8 cameras, allowing cameras with overlapping fields of view to share context. Shape remains `(B, 11, 8, 256)`.

**Stage 4 — Temporal Pooling:** Another learned gated softmax aggregates across the 11 timesteps per camera, masking empty timesteps. This collapses the temporal dimension to produce `(B, 8, 256)` — one 256-dim feature vector per camera.

**Output Projection:** A final linear layer (256 → 256) projects the output. Both its weight and bias are **initialized to zero**, so the camera branch starts as a complete no-op. The encoder also returns a `(B, 8)` boolean mask indicating which cameras have data.

**Key design decisions:**

| Decision | Rationale |
|---|---|
| **Learned gated pooling** (not mean/max) | Variable-length token sequences need attention-weighted aggregation, not position-agnostic pooling |
| **Cross-camera self-attention** | Cameras with overlapping FOV share context; a single transformer layer over 8 tokens is cheap |
| **Zero-initialized output projection** | Camera branch starts as a **no-op** — the model can train stably from a pretrained GameFormer checkpoint without the camera signal disrupting learned features. The camera gradually "turns on" as its weights train. |
| **5x learning rate for camera params** | Compensates for zero-init cold start; camera parameters need to catch up to the already-warm trajectory/map encoders |

# LiDAR Data Processing & Encoding

To effectively integrate raw, continuous 3D LiDAR data into the GameFormer architecture, we employ a structured Bird's Eye View (BEV) representation combined with early temporal fusion. This allows the network to extract dense, low-level kinematics while managing the heavy computational constraints of 3D point clouds.

## Data Preprocessing
Before entering the neural network, the raw LiDAR point clouds undergo several critical transformations:
* **Agent-Centric Coordinate Normalization:** To ensure spatial generalization and rotation/translation invariance, the LiDAR data is transformed from ego-centric coordinates to agent-centric coordinates. By setting the target agent as the origin (0,0) and aligning its heading to the +X axis, we decouple the agent's kinematics from the ego-vehicle's motion. This prevents the network from wasting computational capacity learning to subtract the autonomous vehicle's velocity from the scene.
* **Voxelization (BEV Grids):** Point-based processing across multiple temporal frames exceeds standard computational budgets. We solve this by discretizing the unstructured 3D point cloud into a structured Bird's Eye View spatial grid, creating a uniform format suitable for 3D Convolutions. 
* **Early Temporal Fusion:** Autonomous driving is a 4D spatio-temporal problem. Rather than processing frames individually (which risks kinematic blindness), we stack the temporal sweeps into the channel dimension. This allows the network to extract coupled motion features—like velocity and acceleration—directly from raw geometric displacements.

## LiDAREncoder Architecture
The `LiDAREncoder` is a custom 3-layer 3D Convolutional Neural Network (CNN) designed to process the multi-frame BEV grids and project them into the Transformer's latent space.

* **Spatio-Temporal Extraction:** The network utilizes 3D Convolutional layers with ReLU activations to simultaneously scan across the spatial ($X, Y$) and temporal ($Z$/channel) dimensions.
* **Spatial Decimation:** To strictly manage the parameter count and memory overhead before feeding data to the Transformer, the network employs aggressive spatial downsampling with a spatial stride of `4` across all three convolutional layers. *(Note: This architectural trade-off serves as a massive regularizer for training stability, though it establishes a resolution bottleneck by compressing fine geometric details).*
* **Feature Projection:** Following the convolutions and a `0.2` Dropout layer, the spatial dimensions are flattened, and the features are passed through a Feed-Forward Network to generate 256-dimensional tokens.

## LiDAR - GameFormer Transformer Integration
Once the LiDAR BEV grid is processed by the `LiDAREncoder`, it is integrated into the primary GameFormer Transformer pipeline alongside map and agent data:
* **Dynamic Masking:** Because LiDAR point clouds are inherently sparse, the pipeline calculates a boolean `lidar_mask` by identifying entirely empty spatial bins `(lidar_bev.sum == 0)`. This mask prevents the Transformer's attention mechanism from wasting compute on empty space.
* **Attention Fusion:** The encoded LiDAR tokens are concatenated directly with the encoded actors, map lanes, crosswalks, and camera tokens. This creates a massive multimodal context vector that is passed into the multi-layer Transformer Encoder, allowing the network to cross-attend between the dense physical geometry of the LiDAR and the high-level semantic intents of the vector data.

### Fusion into the Encoder

In `model/GameFormer.py`, the `Encoder` concatenates the 8 camera feature vectors as additional sequence elements alongside the encoded agents, lanes, and crosswalks before passing everything through the 6-layer fusion `TransformerEncoder`. If no camera tokens are available, the model falls back to the original agent + map fusion without any code path changes.

Camera tokens are **scene-level** (shared across all agents), not per-agent. Each agent's encoding attends to the same 8 camera features but different agent-centric map features.

### Training Configuration for Camera

The camera branch has its own optimizer parameter group in `interaction_prediction/train.py`. All parameters belonging to camera-related modules (`camera_encoder`, `camera_cross_attn`, `camera_cross_norm`, `camera_cross_ffn`, `camera_cross_ffn_norm`, `aux_head`) are grouped together and trained with a **5x higher learning rate** (5e-4 vs the base 1e-4) using AdamW.

No separate auxiliary camera loss is used — the camera signal is trained end-to-end through the main trajectory prediction loss (GMM log-likelihood + mode selection cross-entropy).

---

## Original GameFormer Architecture (High-Level)

The base architecture that we build on top of:

### Encoder (`model/GameFormer.py`)

Independently encodes each input modality and fuses them via a 6-layer TransformerEncoder (8 heads, 256-dim):

- **AgentEncoder**: 2-layer LSTM over 11-step trajectory history (x, y, heading, vx, vy, length, width, height) + learned type embedding
- **LaneEncoder**: PointNet-style MLP over lane centerlines, left/right boundaries, speed limits, traffic signals, stop signs
- **CrosswalkEncoder**: 3-layer MLP over crosswalk polygon points

### Decoder (`model/GameFormer.py`)

Multi-level **game-theoretic reasoning** (level-K):

- **Level 0**: Each agent independently decodes 6 multi-modal trajectory predictions via cross-attention
- **Levels 1..K**: Each agent observes the predicted futures of other agents from the previous level, encodes interactions via self-attention, and re-decodes conditioned on this interaction context
- Each mode outputs GMM parameters (per-timestep mean + log-variance) and a mode selection score

### Loss

Hierarchical level-K loss across all decoder levels, combining:
- GMM negative log-likelihood on the best-matching mode
- Cross-entropy mode selection loss with label smoothing (0.2)

---

## Data Processing

`interaction_prediction/data_process.py` converts raw Waymo TFRecord scenarios into preprocessed `.npz` files.

### Usage

Run `data_process.py` with `--load_path` (Waymo TFRecords), `--save_path` (output directory), and optionally `--camera_dir` (separate camera TFRecords). Add `--test` for test data (no ground truth futures) or `--use_multiprocessing --processes 8` for parallel processing.

### Output `.npz` format

| Key | Shape | Description |
|---|---|---|
| `ego` | (2, 11, 9) | History states for both agents |
| `neighbors` | (32, 11, 9) | Surrounding agent histories |
| `map_lanes` | (2, 6, 300, 17) | Lane features per agent |
| `map_crosswalks` | (2, 4, 100, 3) | Crosswalk features per agent |
| `gt_future_states` | (2, 80, 5) | Ground truth futures (train only) |
| **`camera_tokens`** | **(11, 8, 256)** | **Discretized camera tokens (our addition)** |
| **`lidar_bev`** | **(12, 11, 300, 300)** | **BEV LiDAR data (our addition)** |
| `object_type` | (2,) | Agent types (1=vehicle, 2=pedestrian, 3=cyclist) |
| `object_index` | (2,) | Waymo track IDs |
| `current_state` | (4,) | (x, y, z, heading) at t=0 |


---

## Training

Training uses PyTorch Distributed Data Parallel (DDP) via `torchrun`. Point `--train_set` and `--valid_set` to directories of preprocessed `.npz` files.

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 16 | Per-GPU batch size |
| `--training_epochs` | 30 | Total epochs |
| `--learning_rate` | 1e-4 | Base LR (camera uses 5x) |
| `--level` | 3 | Decoder reasoning levels (K) |
| `--modalities` | 6 | Trajectory modes |
| `--future_len` | 80 | Prediction horizon (8s at 10Hz) |
| `--encoder_layers` | 6 | Transformer encoder layers |

**Optimizer**: AdamW, two param groups (base 1e-4, camera 5e-4). **Scheduler**: MultiStepLR at epochs [20, 22, 24, 26, 28], gamma=0.5. **Gradient clip**: norm 5. **Metrics**: ADE, FDE, Waymo motion metrics (minADE, minFDE, miss rate, overlap rate, mAP).

---

## Ablation Experiments

Camera fusion ablations are tracked across branches:

| Branch | Description |
|---|---|
| `version/modalities/base` | Baseline — original GameFormer, no camera |
| `versions/modalities/baseline-plus-lidar` | GameFormer + LiDAR |
| `version/modalities/agent-camera` | GameFormer + Camera (V1) |
| `version/modalities/final` | GameFormer + Camera (V2) |
| `versions/best-performer` | GameFormer + Camera (V3) — best result |


---

## Requirements

Python 3.8+, PyTorch 1.12+ (CUDA), TensorFlow (for Waymo data loading), waymo-open-dataset-tf-2-11-0, shapely, numpy, matplotlib, tqdm. See `requirements.txt`.

## Project Structure

```
GameFusion/
├── model/
│   ├── GameFormer.py          # Encoder-Decoder (camera integration here)
│   └── modules.py             # All building blocks (CameraTokenEncoder here)
├── interaction_prediction/
│   ├── data_process.py        # Data preprocessing (camera extraction here)
│   ├── train.py               # DDP training (camera LR config here)
│   ├── interaction_test.py    # Evaluation
│   └── train.sh               # Launch script
├── open_loop_planning/        # Open-loop trajectory planning
├── utils/
│   ├── data_utils.py          # Map/geometry utilities
│   ├── inter_pred_utils.py    # Loss functions, metrics, dataset class
│   └── ...
└── requirements.txt
```

Acknowledgments: We thank the original GameFormer authors for their open-source codebase, which provided a strong foundation for our camera fusion extension. We also acknowledge the Waymo Open Dataset team for providing the rich data that made this project possible. Additionally, we utilized Generative AI tools for code generation, analysis and debugging assistance, which significantly accelerated our development process.