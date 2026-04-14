import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image


def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
        depth:          [rfn,pn,1]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts, torch.ones([pn, 1], device=pts.device, dtype=pts.dtype)], 1)
    srn = Rt.shape[0]
    KRt = K @ Rt
    last_row = torch.zeros([srn, 1, 4], device=pts.device, dtype=pts.dtype)
    last_row[:, :, 3] = 1.0
    H = torch.cat([KRt, last_row], 1)
    pts_cam = H[:, None, :, :] @ hpts[None, :, :, None]
    pts_cam = pts_cam[:, :, :3, 0]
    depth = pts_cam[:, :, 2:]
    invalid_mask = torch.abs(depth) < 1e-4
    depth = torch.where(invalid_mask, torch.full_like(depth, 1e-3), depth)
    pts_2d = pts_cam[:, :, :2] / depth
    return pts_2d, ~(invalid_mask[..., 0]), depth


def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """
    :param feats:   b,f,h,w
    :param points:  b,n,2
    :return: feats_inter: b,n,f
    """
    _, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)
    return feats_inter.permute(0, 2, 1)


class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32, skip_xmem=True):
        # skip_xmem kept for backward-compat; XMem/detection paths have been removed.
        del skip_xmem
        self.device = device
        self.dtype = dtype
        self.mu = 0.02

        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam

        self.feat_backbone = feat_backbone
        if feat_backbone == 'dinov2':
            self.dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.dinov2_feat_extractor.eval()
            self.dinov2_feat_extractor.to(dtype=self.dtype)
            self._dinov2_on_gpu = False
        elif feat_backbone == 'dinov3':
            from transformers import AutoModel
            dinov3_repo = os.environ.get('DINOV3_REPO', 'facebook/dinov3-vitl16-pretrain-lvd1689m')
            self.dinov3_feat_extractor = AutoModel.from_pretrained(dinov3_repo)
            self.dinov3_feat_extractor.eval()
            self.dinov3_feat_extractor.to(dtype=self.dtype)
            self._dinov3_num_reg = getattr(self.dinov3_feat_extractor.config, 'num_register_tokens', 4)
            self._dinov3_patch = getattr(self.dinov3_feat_extractor.config, 'patch_size', 16)
            self._dinov3_on_gpu = False
        else:
            raise NotImplementedError

    def eval(self, pts, return_names=['dino_feats'], return_inter=False):
        # :param pts: (N, 3) torch tensor in world frame
        # :return: dict with 'dist', 'valid_mask', and each name in return_names
        assert len(self.curr_obs_torch) > 0, 'Please call update() first!'
        assert type(pts) == torch.Tensor
        assert pts.ndim == 2 and pts.shape[1] == 3

        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[..., 0]

        inter_depth = interpolate_feats(
            self.curr_obs_torch['depth'].unsqueeze(1), pts_2d,
            h=self.H, w=self.W, padding_mode='zeros',
            align_corners=True, inter_mode='nearest')[..., 0]

        dist = inter_depth - pts_depth
        dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu)
        dist_weight = torch.exp(torch.clamp(self.mu - torch.abs(dist), max=0) / self.mu)
        dist = torch.clamp(dist, min=-self.mu, max=self.mu)
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6)

        dist_all_invalid = (dist_valid.float().sum(0) == 0)
        dist = torch.where(dist_all_invalid, torch.full_like(dist, 1e3), dist)

        outputs = {'dist': dist, 'valid_mask': ~dist_all_invalid}

        for k in return_names:
            inter_k = interpolate_feats(
                self.curr_obs_torch[k].permute(0, 3, 1, 2), pts_2d,
                h=self.H, w=self.W, padding_mode='zeros',
                align_corners=True, inter_mode='bilinear')
            val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6)
            val = torch.where(dist_all_invalid.unsqueeze(-1), torch.zeros_like(val), val)
            outputs[k] = val
            if return_inter:
                outputs[k + '_inter'] = inter_k
            else:
                del inter_k

        return outputs

    def batch_eval(self, pts, return_names=['dino_feats']):
        batch_pts = 60000
        outputs = {}
        for i in range(0, pts.shape[0], batch_pts):
            out = self.eval(pts[i:i + batch_pts], return_names=return_names)
            for k, v in out.items():
                outputs.setdefault(k, []).append(v)
        for k in outputs:
            if outputs[k][0] is not None:
                outputs[k] = torch.cat(outputs[k], dim=0)
            else:
                outputs[k] = None
        return outputs

    def extract_dinov2_features(self, imgs, params):
        K = imgs.shape[0]
        patch_h, patch_w = params['patch_h'], params['patch_w']
        feat_dim = 1024  # vitl14

        transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=self.device)
        for j in range(K):
            imgs_tensor[j] = transform(Image.fromarray(imgs[j]))[:3]

        with torch.no_grad():
            # Move to GPU for inference, then back to CPU to free VRAM.
            if not self._dinov2_on_gpu:
                self.dinov2_feat_extractor.to(self.device)
                self._dinov2_on_gpu = True
            features = self.dinov2_feat_extractor.forward_features(
                imgs_tensor.to(dtype=self.dtype))['x_norm_patchtokens']
            features = features.reshape((K, patch_h, patch_w, feat_dim))
            self.dinov2_feat_extractor.cpu()
            self._dinov2_on_gpu = False
            torch.cuda.empty_cache()
        return features

    def extract_dinov3_features(self, imgs, params):
        K = imgs.shape[0]
        patch_h, patch_w = params['patch_h'], params['patch_w']
        p = self._dinov3_patch
        feat_dim = self.dinov3_feat_extractor.config.hidden_size

        transform = T.Compose([
            T.Resize((patch_h * p, patch_w * p)),
            T.CenterCrop((patch_h * p, patch_w * p)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        imgs_tensor = torch.zeros((K, 3, patch_h * p, patch_w * p), device=self.device)
        for j in range(K):
            imgs_tensor[j] = transform(Image.fromarray(imgs[j]))[:3]

        with torch.no_grad():
            if not self._dinov3_on_gpu:
                self.dinov3_feat_extractor.to(self.device)
                self._dinov3_on_gpu = True
            out = self.dinov3_feat_extractor(pixel_values=imgs_tensor.to(dtype=self.dtype))
            tokens = out.last_hidden_state
            # Drop CLS + register tokens, keep only patch tokens.
            patch_tokens = tokens[:, 1 + self._dinov3_num_reg:, :]
            features = patch_tokens.reshape((K, patch_h, patch_w, feat_dim))

            nan_mask = ~torch.isfinite(features).all(dim=-1)
            n_bad = int(nan_mask.sum().item())
            if n_bad > 0:
                total = K * patch_h * patch_w
                per_cam = nan_mask.reshape(K, -1).sum(dim=1).tolist()
                print(f"  [dinov3] WARNING: {n_bad}/{total} patch tokens non-finite "
                      f"(per-cam: {per_cam}, dtype={self.dtype}). "
                      "Replacing with zeros.")
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            self.dinov3_feat_extractor.cpu()
            self._dinov3_on_gpu = False
            torch.cuda.empty_cache()
        return features

    def extract_features(self, imgs, params):
        if self.feat_backbone == 'dinov2':
            return self.extract_dinov2_features(imgs, params)
        if self.feat_backbone == 'dinov3':
            return self.extract_dinov3_features(imgs, params)
        raise NotImplementedError

    def update(self, obs):
        # :param obs: dict with 'color' (K,H,W,3) np, 'depth' (K,H,W) np,
        #             'pose' (K,4,4) np, 'K' (K,3,3) np
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {'patch_h': color.shape[1] // 10, 'patch_w': color.shape[2] // 10}

        self.curr_obs_torch['dino_feats'] = self.extract_features(color, params)
        self.curr_obs_torch['color'] = color
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=self.dtype) / 255.0
        # Sensor depth frequently contains NaN/Inf for missing returns.
        # Map them to 0 so the `inter_depth > 0` check in eval() correctly
        # excludes those pixels; otherwise NaNs poison dist_weight and the
        # aggregated feature (since 0 * NaN = NaN under IEEE 754).
        depth_np = np.where(np.isfinite(obs['depth']), obs['depth'], 0.0)
        self.curr_obs_torch['depth'] = torch.from_numpy(depth_np).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=self.dtype)

        _, self.H, self.W = obs['depth'].shape
