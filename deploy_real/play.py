import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from unitree_sdk2py.utils.crc import CRC

from go2_cfg import Go2Cfg
from common.remote_controller import RemoteController, KeyMap
from common.command_helper import init_cmd, create_zero_cmd, create_damping_cmd
from common.rotation_helper import get_gravity_orientation


class Controller:
    def __init__(self, cfg: Go2Cfg):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("[policy device]: ", self.device)

        # load policy
        self.policy = torch.jit.load(self.cfg.policy_path).to(self.device)
        self.policy.eval()
        self._warm_up()
        # load observation
        self.obs = np.zeros(self.cfg.num_obs, dtype=np.float32)
        self.action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.base_ang_vel_history = np.zeros(3 * self.cfg.num_history, dtype=np.float32)
        self.projected_gravity_history = np.zeros_like(self.base_ang_vel_history, dtype=np.float32)
        self.q_history = np.zeros(12 * self.cfg.num_history, dtype=np.float32)
        self.dq_history = np.zeros_like(self.q_history, dtype=np.float32)
        self.action_history = np.zeros_like(self.q_history, dtype=np.float32)
        self.rl_start_flag: bool = False
        # load unitree sdk
        self.qj = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.cmd = self.cfg.cmd.copy()
        self.remote_controller = RemoteController()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher = ChannelPublisher(self.cfg.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(self.cfg.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self._wait_for_low_state()
        init_cmd(self.low_cmd)

    def _warm_up(self):
        obs = torch.zeros(self.cfg.num_obs, dtype=torch.float32, device=self.device)
        for _ in range(10):
            _ = self.policy(obs)
        print("Policy has been warmed up.")

    def _wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.cfg.control_dt)
        print("Successfully connected to the robot.")

    def _low_state_handler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt)

    def move_to_default_pos(self):
        print('Moving to default pos.')
        total_time = 2
        num_step = int(total_time / self.cfg.control_dt)

        init_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.low_state.motor_state[i].q

        for step in range(num_step):
            alpha = step / num_step
            for i in range(12):
                target_pos = self.cfg.default_angles[i]
                self.low_cmd.motor_cmd[i].q = init_dof_pos[i] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[i].dq = 0.0  # qd
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 0.6
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.cfg.default_angles[i]
                self.low_cmd.motor_cmd[i].dq = 0.0  # qd
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 0.6
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt)

    def run(self):
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        quat = np.array([self.low_state.imu_state.quaternion], dtype=np.float32)
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        projected_gravity = get_gravity_orientation(quat)  # imu_state quaternion: w, x, y, z

        q = self.qj.copy()
        dq = self.dqj.copy()

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
            self.base_ang_vel_history[-3:] = ang_vel * self.cfg.ang_vel_scale
            self.projected_gravity_history[-3:] = projected_gravity
            self.q_history[-12:] = q - self.cfg.default_angles
            self.dq_history[-12:] = dq * self.cfg.dof_vel_scale
            self.action_history[-12:] = self.action

        obs_history = np.concatenate([self.base_ang_vel_history,
                                      self.projected_gravity_history,
                                      self.q_history,
                                      self.dq_history,
                                      self.action_history])
        self.obs[:3] = self.cmd
        self.obs[3:] = obs_history

        obs_tensor = torch.from_numpy(self.obs).to(self.device)
        self.action[:] = self.policy(obs_tensor).cpu().detach().numpy()

        target_q = self.cfg.default_angles
        # target_q = self.cfg.default_angles + self.cfg.action_scale * self.action

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = target_q[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.cfg.kps[i]
            self.low_cmd.motor_cmd[i].kd = self.cfg.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.cfg.control_dt)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="network interface")
    args = parser.parse_args()

    go2_cfg = Go2Cfg()

    ChannelFactoryInitialize(0, args.net)
    controller = Controller(go2_cfg)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print('Exit')
