from dataclasses import dataclass

import numpy as np


@dataclass
class Go2Cfg:
    xml_path: str = "assets/mujoco/unitree_robots/go2/scene.xml"

    log_enable: bool = True
    # policy_path: str = "policy/flat/1006-1.pt"
    # policy_path: str = "policy/rough/1014-1.pt"
    # logger_type: str = "our_flat"
    # policy_path: str = "policy/rough/1107-1-s.pt"
    # logger_type: str = "our_flat_slim"
    policy_path: str = "policy/wo_contact/rough/1108-1-s.pt"
    logger_type: str = "our_wo_contact_flat_slim"
    # policy_path: str = "policy/dreamwaq/rough/1015-1.pt"
    # logger_type: str = "dreamwaq_flat"

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

    cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
