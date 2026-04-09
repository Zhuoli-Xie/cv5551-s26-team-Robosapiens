import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import os
import json

# ============================================
# CAMERA PARAMETERS (ZED2i - Load from your calibration)
# ============================================
class CameraParameters:
    def __init__(self, name, image_path, depth_path, intrinsics, extrinsics):
        self.name = name
        self.image_path = image_path      # RGB image (cam1.png, cam2.png)
        self.depth_path = depth_path      # Depth map from ZED2i
        self.intrinsics = intrinsics      # 3x3 camera matrix
        self.extrinsics = extrinsics      # 4x4 world to camera transform
        self.width = None
        self.height = None

def load_camera_calibration():
    """
    Load actual ZED2i calibration data
    Replace paths with your actual calibration files
    """
    with open('zed2i_calibration.json', 'r') as f:
        calib = json.load(f)
    
    intrinsics_cam1 = np.array(calib['cam1']['intrinsics'])
    extrinsics_cam1 = np.array(calib['cam1']['extrinsics'])  # World to cam1
    
    intrinsics_cam2 = np.array(calib['cam2']['intrinsics'])
    extrinsics_cam2 = np.array(calib['cam2']['extrinsics'])  # World to cam2
    
    cam1 = CameraParameters(
        name="cam1",
        image_path="cam1.png",
        depth_path="cam1_depth.npy",
        intrinsics=intrinsics_cam1,
        extrinsics=extrinsics_cam1
    )
    
    cam2 = CameraParameters(
        name="cam2",
        image_path="cam2.png",
        depth_path="cam2_depth.npy",
        intrinsics=intrinsics_cam2,
        extrinsics=extrinsics_cam2
    )
    
    return [cam1, cam2]

# ============================================
# 1. LOAD MODELS
# ============================================
print("="*50)
print("LOADING MODELS")
print("="*50)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Grounding DINO
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
grounding_dino.eval().to(device)

# DINOv2
print("Loading DINOv2 ViT-S/14...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2.eval().to(device)

# SAM
sam_checkpoint = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam = sam.to(device)
sam_predictor = SamPredictor(sam)

print("All models loaded\n")

# ============================================
# 2. BUILD W: PER-CAMERA FEATURE VOLUMES
# ============================================
class D3Fields:
    """
    D^3 Fields: Dynamic 3D Descriptor Fields
    Implements F(x | W) from the paper
    W = { (R_i, W^f_i, W^p_i) } for each camera i
    """
    
    def __init__(self, cameras, truncation_mu=0.1):
        self.cameras = cameras
        self.mu = truncation_mu  # Truncation threshold for TSDF (meters)
        self.delta = 1e-6
        
        # Store per-camera volumes
        self.depth_maps = []           # R_i: Depth maps from ZED2i
        self.semantic_features = []    # W^f_i: DINOv2 features
        self.instance_masks = []       # W^p_i: Instance masks from Grounded-SAM
        
        self._build_feature_volumes()
    
    def _build_feature_volumes(self):
        """Build W for all cameras"""
        
        for cam in self.cameras:
            print(f"\nProcessing {cam.name}...")
            
            # Load RGB image
            image_pil = Image.open(cam.image_path)
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            cam.width, cam.height = image_pil.size
            
            # Convert to numpy for SAM
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # ============================================
            # R_i: Load depth map from ZED2i (already exists)
            # ============================================
            depth_map = np.load(cam.depth_path)
            self.depth_maps.append(depth_map)
            print(f"   Depth map R_i loaded: {depth_map.shape}")
            
            # ============================================
            # W^f_i: DINOv2 dense patch feature map
            # FIX 1: Use forward_features + patch tokens instead of
            # forward(), which returns only the [CLS] token vector.
            # The paper requires W^f_i in R^{H x W x N} (spatial map).
            # ============================================
            semantic_features = self._extract_dinov2_features(image_pil)
            self.semantic_features.append(semantic_features)
            print(f"   Semantic features W^f_i: {semantic_features.shape}")
            
            # ============================================
            # W^p_i: Instance masks from Grounded-SAM
            # ============================================
            instance_masks = self._extract_instance_masks(image_pil, image_np)
            self.instance_masks.append(instance_masks)
            print(f"   Instance masks W^p_i: {instance_masks.shape}")
    
    def _extract_dinov2_features(self, image_pil):
        """
        Extract DINOv2 dense spatial feature map W^f_i.
        Returns: (H, W, 384) feature volume.

        FIX 1: dinov2(img) returns the [CLS] token — a single (1, 384) vector,
        not a spatial map. We use forward_features() and take x_norm_patchtokens
        to get one embedding per patch, then reshape to (patch_h, patch_w, 384)
        and resize to the full image resolution.
        """
        from torchvision import transforms
        
        # ViT-S/14 with 224px input → 16x16 = 256 patches
        input_size = 224
        patch_size = 14
        patch_grid = input_size // patch_size  # 16

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = dinov2.forward_features(img_tensor)
            # x_norm_patchtokens: (1, num_patches, 384) — spatial patch embeddings
            features = out["x_norm_patchtokens"]  # (1, 256, 384)
        
        features = features.squeeze(0).cpu().numpy()          # (256, 384)
        features = features.reshape(patch_grid, patch_grid, -1)  # (16, 16, 384)
        
        # Resize back to original image resolution
        h, w = image_pil.size[1], image_pil.size[0]
        features_resized = cv2.resize(features, (w, h), interpolation=cv2.INTER_LINEAR)
        # cv2.resize drops the channel dim when C==1, but 384 > 1 so shape stays (H, W, 384)
        
        return features_resized  # (H, W, 384)
    
    def _extract_instance_masks(self, image_pil, image_np):
        """
        Extract instance masks W^p_i using Grounded-SAM.
        Returns: (H, W, M) where M = number of instances + 1 (background)
        """
        text = "object. phone. remote. cup. book. shoe. pen. mug. fork. spoon."
        inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = grounding_dino(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[(image_pil.size[1], image_pil.size[0])],
            text_labels=True
        )
        
        sam_predictor.set_image(image_np)
        
        M = len(results[0]["boxes"]) + 1  # +1 for background
        instance_volume = np.zeros((image_pil.size[1], image_pil.size[0], M), dtype=np.float32)
        instance_volume[:, :, 0] = 1.0   # background channel
        
        for idx, (box, label, score) in enumerate(zip(
            results[0]["boxes"], results[0]["labels"], results[0]["scores"]
        )):
            box_np = box.cpu().numpy()
            masks, mask_scores, _ = sam_predictor.predict(
                box=box_np,
                multimask_output=False
            )
            mask = masks[0]
            instance_volume[mask, idx + 1] = 1.0
            instance_volume[mask, 0] = 0.0   # remove from background
            print(f"      Instance {idx+1}: {label} (score: {score:.3f})")
        
        return instance_volume  # (H, W, M)
    
    # ============================================
    # 3. F(x | W): Query arbitrary 3D points
    # ============================================
    def query(self, x_world):
        """
        Implements F(x | W) from Equations 2-6.

        Args:
            x_world: (N, 3) array or tensor of 3D points in world coordinates.

        Returns:
            d: (N,)      signed distances (Eq. 6, numerator uses v_i * d'_i)
            f: (N, 384)  semantic descriptors (Eq. 6, numerator uses v_i * w_i * f_i)
            p: (N, M)    instance probabilities (Eq. 6, numerator uses v_i * w_i * p_i)
        """
        N = x_world.shape[0]
        
        if not isinstance(x_world, torch.Tensor):
            x_world = torch.tensor(x_world, dtype=torch.float32, device=device)
        else:
            x_world = x_world.to(device)
        
        M = self.instance_masks[0].shape[-1]
        
        d_total      = torch.zeros(N,       device=device)
        f_total      = torch.zeros(N, 384,  device=device)
        p_total      = torch.zeros(N, M,    device=device)
        # FIX 3: d is weighted by Σ v_i; f and p are weighted by Σ v_i * w_i
        v_total      = torch.zeros(N,       device=device)   # denominator for d
        vw_total     = torch.zeros(N,       device=device)   # denominator for f, p
        
        for cam_idx, cam in enumerate(self.cameras):
            # FIX 4: convert extrinsics to tensor on the correct device inside helper
            points_cam = self._world_to_camera(x_world, cam.extrinsics)
            u, v_coord, depths = self._camera_to_image(points_cam, cam.intrinsics)
            
            valid = (u >= 0) & (u < cam.width) & (v_coord >= 0) & (v_coord < cam.height) & (depths > 0)
            
            if not valid.any():
                continue
            
            u_valid = u[valid].long()
            v_valid = v_coord[valid].long()
            
            # Depth reading r'_i from depth map (Eq. 3)
            depth_map = torch.tensor(self.depth_maps[cam_idx], dtype=torch.float32, device=device)
            r_prime = depth_map[v_valid, u_valid]
            
            # Truncated signed distance d_i and d'_i (Eq. 3)
            d_i       = depths[valid] - r_prime
            d_i_trunc = torch.clamp(d_i, -self.mu, self.mu)
            
            # Visibility v_i and weight w_i (Eq. 4)
            v_i = (d_i < self.mu).float()

            # FIX 2: clamp with max=0, not min=0.
            # When |d_i| > mu the exponent must be negative (decaying weight).
            # Original code used clamp(min=0) which forced w_i = exp(0) = 1 always.
            w_i = torch.exp(torch.clamp(self.mu - torch.abs(d_i), max=0.0) / self.mu)
            
            # Semantic features f_i and instance mask p_i (Eq. 5)
            semantic_volume  = torch.tensor(self.semantic_features[cam_idx],
                                            dtype=torch.float32, device=device)
            instance_volume  = torch.tensor(self.instance_masks[cam_idx],
                                            dtype=torch.float32, device=device)
            f_i = semantic_volume[v_valid, u_valid, :]   # (N_valid, 384)
            p_i = instance_volume[v_valid, u_valid, :]   # (N_valid, M)
            
            # Accumulate (Eq. 6)
            d_total[valid]  += v_i * d_i_trunc
            f_total[valid]  += (v_i * w_i).unsqueeze(-1) * f_i
            p_total[valid]  += (v_i * w_i).unsqueeze(-1) * p_i
            v_total[valid]  += v_i
            vw_total[valid] += v_i * w_i
        
        # Normalize (Eq. 6): d uses Σv_i, f and p use Σ(v_i * w_i)
        has_v  = v_total  > 0
        has_vw = vw_total > 0

        d_total[has_v]  = d_total[has_v]  / (self.delta + v_total[has_v])
        f_total[has_vw] = f_total[has_vw] / (self.delta + vw_total[has_vw].unsqueeze(-1))
        p_total[has_vw] = p_total[has_vw] / (self.delta + vw_total[has_vw].unsqueeze(-1))
        
        return d_total, f_total, p_total
    
    def _world_to_camera(self, points_world, extrinsics):
        """
        Transform points from world to camera coordinates.

        FIX 4: Convert the numpy extrinsics array to a float32 tensor on the
        same device as points_world before doing any arithmetic. The original
        code left extrinsics as numpy, causing a device/type mismatch at runtime.
        """
        ext = torch.tensor(extrinsics, dtype=torch.float32, device=points_world.device)
        R   = ext[:3, :3]
        t   = ext[:3, 3]
        return points_world @ R.T + t.unsqueeze(0)
    
    def _camera_to_image(self, points_cam, intrinsics):
        """Project camera-space points to image pixel coordinates."""
        intr = torch.tensor(intrinsics, dtype=torch.float32, device=points_cam.device)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        
        depths = points_cam[:, 2]
        u = (points_cam[:, 0] * fx / depths) + cx
        v = (points_cam[:, 1] * fy / depths) + cy
        
        return u, v, depths


# ============================================
# 4. USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    cameras = load_camera_calibration()
    
    d3_fields = D3Fields(cameras, truncation_mu=0.1)
    
    x_world = torch.tensor([
        [0.0,  0.0,  1.0],
        [0.1,  0.1,  0.9],
        [-0.1, 0.05, 1.1],
    ], device=device)
    
    distances, semantic_features, instance_probs = d3_fields.query(x_world)
    
    print("\n" + "="*50)
    print("QUERY RESULTS")
    print("="*50)
    print(f"Signed distances:           {distances}")
    print(f"Semantic features shape:    {semantic_features.shape}")
    print(f"Instance probabilities shape: {instance_probs.shape}")


'''
Example Calibration File needed
{
  "cam1": {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]]
  },
  "cam2": { ... }
}

need to get depth maps from both images


'''
