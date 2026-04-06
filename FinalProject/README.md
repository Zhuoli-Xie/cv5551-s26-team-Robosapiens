# D<sup>3</sup>Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Rearrangement

# create conda environment
mamba env create -f env.yaml
conda activate d3fields


# download pretrained models
bash scripts/download_ckpts.sh

data path: FinalProject/mugs

Optimization file: grasp_pose_search.py