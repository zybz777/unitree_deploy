from dataclasses import dataclass

import numpy as np


@dataclass
class Go2Cfg:
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_boxes.xml"

    log_enable: bool = True
    # flat
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_flat_slim"
    # policy_path: str = "policy/wo_contact/rough/1108-1-s.pt"
    # logger_type: str = "our_wo_contact_flat_slim"
    # policy_path: str = "policy/wo_fusion/rough/1109-1-s.pt"
    # logger_type: str = "our_wo_fusion_flat_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_flat"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_flat"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_flat"
    # stair5
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_stair5.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_stair5_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_stair5"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_stair5"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_stair5"
    # stair10
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_stair10.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_stair10_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_stair10"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_stair10"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_stair10"
    # stair15
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_stair15.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_stair15_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_stair15"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_stair15"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_stair15"
    # slope10
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_slope10.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_slope10_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_slope10"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_slope10"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_slope10"
    # slope20
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_slope20.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_slope20_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_slope20"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_slope20"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_slope20"
    # slope30
    xml_path: str = "assets/mujoco/unitree_robots/go2/scene_slope30.xml"
    policy_path: str = "policy/rough/1118-1-s.pt"
    logger_type: str = "our_slope30_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_slope30"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_slope30"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_slope30"
    # boxes
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_boxes.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_boxes_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_boxes"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_boxes"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_boxes"
    # random rough
    # xml_path: str = "assets/mujoco/unitree_robots/go2/scene_rough.xml"
    # policy_path: str = "policy/rough/1118-1-s.pt"
    # logger_type: str = "our_random_rough_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_random_rough"
    # policy_path: str = "policy/baseline/rough/1114-1.pt"
    # logger_type: str = "baseline_random_rough"
    # policy_path: str = "policy/estimator/rough/1121-1.pt"
    # logger_type: str = "estimator_random_rough"
    # policy_path: str = "policy/wo_contact/rough/1108-1-s.pt"
    # logger_type: str = "our_wo_contact_random_rough_slim"
    # policy_path: str = "policy/wo_fusion/rough/1109-1-s.pt"
    # logger_type: str = "our_wo_fusion_random_rough_slim"

    simulation_duration: float = 600.0
    simulation_dt: float = 0.005
    control_decimation: int = 4

    kps = np.array([25, 25, 25, 25,
                    25, 25, 25, 25,
                    25, 25, 25, 25], dtype=np.float32)
    kds = np.array([0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    default_angles = np.array([-0.1, 0.8, -1.5,
                               0.1, 0.8, -1.5,
                               -0.1, 0.8, -1.5,
                               0.1, 0.8, -1.5], dtype=np.float32)

    ang_vel_scale: float = 0.2
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05

    action_scale: float = 0.25

    num_actions: int = 12
    num_obs: int = 423
    num_history: int = 10
    # baseline
    # num_obs: int = 42+3
    # num_history: int = 1

    cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
