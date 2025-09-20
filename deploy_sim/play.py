import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from go2_cfg import Go2Cfg


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class Env:
    def __init__(self, cfg: Go2Cfg):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load mujoco
        self.m = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.m.opt.timestep = self.cfg.simulation_dt
        self.d = mujoco.MjData(self.m)
        # load policy
        self.policy = torch.jit.load(self.cfg.policy_path).to(self.device)
        self.policy.eval()
        # load observation
        self.obs = np.zeros(self.cfg.num_obs, dtype=np.float32)
        self.action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.target_dof_pos = np.zeros_like(self.action, dtype=np.float32)
        self.base_ang_vel_history = np.zeros(3 * self.cfg.num_history, dtype=np.float32)
        self.projected_gravity_history = np.zeros_like(self.base_ang_vel_history, dtype=np.float32)
        self.q_history = np.zeros(12 * self.cfg.num_history, dtype=np.float32)
        self.dq_history = np.zeros_like(self.q_history, dtype=np.float32)
        self.action_history = np.zeros_like(self.q_history, dtype=np.float32)
        self.rl_start_flag: bool = False

    def get_q(self):
        q = np.array([
            self.d.sensor("FR_hip_pos").data[0], self.d.sensor("FR_thigh_pos").data[0],
            self.d.sensor("FR_calf_pos").data[0],
            self.d.sensor("FL_hip_pos").data[0], self.d.sensor("FL_thigh_pos").data[0],
            self.d.sensor("FL_calf_pos").data[0],
            self.d.sensor("RR_hip_pos").data[0], self.d.sensor("RR_thigh_pos").data[0],
            self.d.sensor("RR_calf_pos").data[0],
            self.d.sensor("RL_hip_pos").data[0], self.d.sensor("RL_thigh_pos").data[0],
            self.d.sensor("RL_calf_pos").data[0],
        ])
        return q

    def get_dq(self):
        dq = np.array([
            self.d.sensor("FR_hip_vel").data[0], self.d.sensor("FR_thigh_vel").data[0],
            self.d.sensor("FR_calf_vel").data[0],
            self.d.sensor("FL_hip_vel").data[0], self.d.sensor("FL_thigh_vel").data[0],
            self.d.sensor("FL_calf_vel").data[0],
            self.d.sensor("RR_hip_vel").data[0], self.d.sensor("RR_thigh_vel").data[0],
            self.d.sensor("RR_calf_vel").data[0],
            self.d.sensor("RL_hip_vel").data[0], self.d.sensor("RL_thigh_vel").data[0],
            self.d.sensor("RL_calf_vel").data[0],
        ])
        return dq

    def key_callback(self, key):
        if key == 265:
            self.cfg.cmd[0] += 0.1
        elif key == 264:
            self.cfg.cmd[0] -= 0.1

        if key == 263:
            self.cfg.cmd[2] += 0.1
        elif key == 262:
            self.cfg.cmd[2] -= 0.1

        if key == 32:
            self.cfg.cmd[:] = 0.
        print(self.cfg.cmd)

    def run(self):
        with mujoco.viewer.launch_passive(self.m, self.d, key_callback=self.key_callback) as viewer:
            start_time = time.time()
            step_dt = self.cfg.control_decimation * self.cfg.simulation_dt
            while viewer.is_running() and time.time() - start_time < self.cfg.simulation_duration:
                step_start = time.time()

                quat = self.d.sensor("imu_quat").data
                ang_vel = self.d.sensor("imu_gyro").data
                projected_gravity = get_gravity_orientation(quat)

                q = self.get_q()
                dq = self.get_dq()

                if not self.rl_start_flag:
                    self.base_ang_vel_history = np.tile(ang_vel * self.cfg.ang_vel_scale, self.cfg.num_history)
                    self.projected_gravity_history = np.tile(projected_gravity, self.cfg.num_history)
                    self.q_history = np.tile(q - self.cfg.default_angles, self.cfg.num_history)
                    self.dq_history = np.tile(dq * self.cfg.dof_vel_scale, self.cfg.num_history)
                    self.action_history = np.tile(self.action, self.cfg.num_history)
                    self.rl_start_flag = True
                else:
                    self.base_ang_vel_history = np.roll(self.base_ang_vel_history, -3)
                    self.projected_gravity_history = np.roll(self.projected_gravity_history, -3)
                    self.q_history = np.roll(self.q_history, -12)
                    self.dq_history = np.roll(self.dq_history, -12)
                    self.action_history = np.roll(self.action_history, -12)
                    self.base_ang_vel_history[-3:] = ang_vel * cfg.ang_vel_scale
                    self.projected_gravity_history[-3:] = projected_gravity
                    self.q_history[-12:] = q - cfg.default_angles
                    self.dq_history[-12:] = dq * cfg.dof_vel_scale
                    self.action_history[-12:] = self.action

                obs_history = np.concatenate([self.base_ang_vel_history,
                                              self.projected_gravity_history,
                                              self.q_history,
                                              self.dq_history,
                                              self.action_history])
                self.obs[:3] = self.cfg.cmd
                self.obs[3:] = obs_history

                obs_tensor = torch.from_numpy(self.obs).to(self.device)
                self.action[:] = self.policy(obs_tensor).cpu().detach().numpy()[0]

                target_q = self.cfg.default_angles + self.cfg.action_scale * self.action

                for _ in range(self.cfg.control_decimation):
                    q = self.get_q()
                    dq = self.get_dq()
                    tau = pd_control(target_q, q, self.cfg.kps, 0, dq, self.cfg.kds)
                    self.d.ctrl = tau
                    mujoco.mj_step(self.m, self.d)

                viewer.sync()

                time_until_next_step = step_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == '__main__':
    cfg = Go2Cfg()
    env = Env(cfg)
    env.run()
