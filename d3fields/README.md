# D3Fields

Grasp-pose transfer for xArm Lite6 with ZED cameras, built on D3Fields descriptors
(GroundingDINO + SAM + DINOv2 fused multi-view point clouds). Given a recorded
grasp on a reference object, the pipeline searches for the equivalent grasp pose
on a novel instance of the same category.

## Installation

### 1. Conda environment

```bash
conda env create -f env.yaml
conda activate d3fields
```

If the `pytorch3d` channel install fails on your CUDA/PyTorch combo, fall back to:

```bash
python scripts/install_pytorch3d.py
```

Additional runtime dependencies not covered by `env.yaml`:

- `pyzed` — install the ZED SDK and its Python API manually from Stereolabs.
- `xarm-python-sdk` — for xArm Lite6 control (`pip install xarm-python-sdk`).
- `pupil-apriltags` — for calibration (`pip install pupil-apriltags`).
- `rerun-sdk`, `plotly`, `scikit-learn` — for visualization in
  `TestBeforeOptimization.py` / `GraspPoseSearch.py`.

### 2. Checkpoints

Download SAM + GroundingDINO weights into `ckpts/`:

```bash
bash scripts/download_ckpts.sh
```

This produces:

```
ckpts/
├── sam_vit_h_4b8939.pth            # SAM ViT-H
├── sam_vit_b_01ec64.pth            # SAM ViT-B (optional, smaller)
├── groundingdino_swint_ogc.pth     # GroundingDINO SwinT
└── groundingdino_swinb_cogcoor.pth # GroundingDINO SwinB
```

DINOv2 weights are fetched automatically via `torch.hub` on first use.

## Execution Order

Run the scripts in this order. Each step writes into a scene directory under
`data/<scene_name>/` that the next step consumes.

### 1. `CameraCalibration.py`

Calibrates each ZED camera against the xArm base frame using four AprilTags on
the table, then saves intrinsics (`camera_params.npy`) and the camera→robot
extrinsic (`camera_extrinsics.npy`) per camera.

```bash
python CameraCalibration.py -o data/<scene>          # auto-detect all ZEDs
python CameraCalibration.py -o data/<scene> --ids 0 1
```

### 2. `RecordGraspPose.py`

Teleoperated / lead-through recording of reference grasp poses. Press
`s` to capture a RGB-D snapshot from each ZED, `Enter` to save the current
xArm TCP pose as a 4×4 matrix (metres), `q` to quit. Also writes the raw
`robot_state.json` for each recorded pose.

```bash
python RecordGraspPose.py --robot-ip <IP> -o data/<scene> --camera-ids 0 1
```

### 3. `TestBeforeOptimization.py`

Sanity-check the perception stack *before* running the (slow) optimization:

1. Runs GroundingDINO + SAM to produce masks, then DINOv2 to produce per-pixel
   features — saved to `<scene>/camera_*/mask/` and `dino_feat/`.
2. Fuses multi-view depth into a merged point cloud and renders it in Open3D.
3. Overlays the recorded gripper poses on the point cloud in an interactive
   Plotly view to verify the gripper geometry / TCP offset is correct.

```bash
python TestBeforeOptimization.py -d data/<scene> --text "bottle"
python TestBeforeOptimization.py -d data/<scene> --skip-detect --pose-idx 0 1 2
```

### 4. `GraspPoseSearch.py`

The main grasp-transfer routine:

1. Loads the reference scene (camera data + masks + D3Fields fusion).
2. Builds a contact set on the reference gripper and queries its descriptors.
3. Runs coarse matching (DINO-feature NN + Procrustes SVD) on the target scene.
4. Refines with a multi-start SE(3) optimization on the grasp cost function.
5. Saves `optimized_grasp.npz` and streams the trajectory to Rerun.

```bash
# Self-test (reference == target)
python GraspPoseSearch.py -d data/<ref_scene> \
    --gripper-pose data/<ref_scene>/target_grasp_pose/pose_0000/gripper_pose.npy

# Transfer to a new object instance
python GraspPoseSearch.py -d data/<ref_scene> --new-scene data/<new_scene> \
    --gripper-pose data/<ref_scene>/target_grasp_pose/pose_0000/gripper_pose.npy
```

## Data Folder Layout

Each scene lives in its own directory under `data/`. After a full run you get:

```
data/<scene>/
├── camera_0/
│   ├── camera_params.npy        # [fx, fy, cx, cy] intrinsics
│   ├── camera_extrinsics.npy    # 4×4 camera→robot extrinsic
│   ├── color/
│   │   └── <t>.png              # RGB frames per timestep
│   ├── depth/
│   │   ├── <t>.npz              # float32 depth in metres
│   │   └── <t>_vis.png          # INFERNO-colormapped preview
│   ├── mask/
│   │   └── <t>.png              # GroundingDINO + SAM mask
│   └── dino_feat/
│       └── <t>.npy              # per-pixel DINOv2 features
├── camera_1/
│   └── ...                      # same layout
├── target_grasp_pose/
│   ├── meta.json                # index + timestamps of recorded poses
│   └── pose_0000/
│       ├── gripper_pose.npy     # 4×4 TCP pose (metres)
│       ├── gripper_pose.txt     # human-readable version
│       └── robot_state.json     # raw xArm state at capture time
└── optimized_grasp.npz          # written by GraspPoseSearch.py
```

Produced by stage:

- **Stage 1 (`CameraCalibration.py`):** `camera_*/camera_params.npy`,
  `camera_*/camera_extrinsics.npy`.
- **Stage 2 (`RecordGraspPose.py`):** `camera_*/color/`, `camera_*/depth/`,
  `target_grasp_pose/`.
- **Stage 3 (`TestBeforeOptimization.py`):** `camera_*/mask/`,
  `camera_*/dino_feat/`.
- **Stage 4 (`GraspPoseSearch.py`):** `optimized_grasp.npz` and, at the data
  root, `grasp_opt.rrd` for replay in Rerun.

## Repository Layout

- `GraphSearch/` — core D3Fields search library (data loading, fusion, contact
  set, coarse matching, SE(3) optimization, visualization).
- `XarmMove/` — xArm Lite6 motion helpers (`move_to_xyzrpy`, `grasp_from_pose`,
  AprilTag cube grasp demo).
- `utils/` — `zed_camera.py` wrapper, `grounded_sam.py`, general `my_utils.py`
  and `draw_utils.py`.
- `scripts/` — checkpoint / dataset download helpers + pytorch3d fallback
  installer.
