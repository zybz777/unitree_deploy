from dataclasses import dataclass

import numpy as np


@dataclass
class Go2Cfg:
    policy_path: str = "policy/flat/0920-1.pt"

    control_dt: float = 0.02

    msg_type: str = "go"

    imu_type: str = "torso"

    lowcmd_topic: str = "rt/lowcmd"
    lowstate_topic: str = "rt/lowstate"

    kps = np.array([25, 25, 25, 25,
                    25, 25, 25, 25,
                    25, 25, 25, 25])
    kds = np.array([0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5])

    default_angles = np.array([-0.1, 0.8, -1.5,
                               0.1, 0.8, -1.5,
                               -0.1, 0.8, -1.5,
                               0.1, 0.8, -1.5])

    ang_vel_scale: float = 0.2
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05

    action_scale: float = 0.25

    num_actions: int = 12
    num_obs: int = 213
    num_history: int = 5

    cmd = np.array([0, 0, 0], dtype=np.float32)
