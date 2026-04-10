"""
grasp_search — Modular grasp pose search via D3Fields descriptor matching.

Modules
-------
    data_loading     : Load scene data, masks, gripper poses, build point clouds
    contact_set      : Construct the contact set D = {(q_i, f_i*)}
    coarse_matching  : DINO feature NN matching -> initial SE(3) alignment
    cost_functions   : Per-term cost components (designed for easy swapping)
    optimization     : Multi-start SE(3) optimizer
    visualization    : Open3D (static) and Rerun (optimization trajectory)
    se3_utils        : Axis-angle / rotation matrix conversions
    evaluation       : Pose error metrics
"""
