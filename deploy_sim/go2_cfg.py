from dataclasses import dataclass

import numpy as np


@dataclass
class Go2Cfg:
    xml_path: str = "assets/mujoco/unitree_robots/go2/scene.xml"

    policy_path: str = "policy/flat/0920-1.pt"

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
    num_obs: int = 213
    num_history: int = 5

    cmd = np.array([0, 0, 0], dtype=np.float32)
