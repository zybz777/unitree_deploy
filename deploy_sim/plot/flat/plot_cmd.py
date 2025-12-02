import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file_path = 'logs_csv/our_flat_slim/obs.csv'
df = pd.read_csv(file_path)

# 起点终点（单位：step）
START_STEP = 250
END_STEP = 500

# 排序（非常重要）
df = df.sort_values('step').reset_index(drop=True)

# ========= 截取区间 =========
mask = (df['step'] >= START_STEP) & (df['step'] <= END_STEP)
data = df.loc[mask].copy()

# ========= 计算时间轴（ms）=========
# 50Hz → 1 step = 20ms
data['timestep_ms'] = (data['step'] - START_STEP) * 20.0

# ========= 绘制一张图：cmd_0 与 lin_vel_0 =========
plt.figure(figsize=(12, 4))

plt.plot(data['timestep_ms'], data['cmd_0'],  label='command vx')
plt.plot(data['timestep_ms'], data['lin_vel_0'], label='actual vx')

plt.xlabel('Timestep [ms]')
plt.ylabel('Tracking Error')
# plt.title('cmd_0 vs lin_vel_0')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('vx_cmd_vs_actual.jpg', dpi=300)
# plt.show()
# ========= 将第二幅图和第三幅图合成一张图（1 行 2 列）=========

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ---------------- 左子图：cmd_1 vs lin_vel_1 ----------------
ax = axes[0]
ax.plot(data['timestep_ms'], data['cmd_1'],  label='command vy')
ax.plot(data['timestep_ms'], data['lin_vel_1'], label='actual vy')
ax.set_xlabel('Timestep [ms]')
ax.set_ylabel('Tracking Error')
ax.grid(True)
ax.legend()
# ax.set_title('vy tracking')

# ---------------- 右子图：cmd_2 vs ang_vel_2 ----------------
ax = axes[1]
ax.plot(data['timestep_ms'], data['cmd_2'],  label='command wz')
ax.plot(data['timestep_ms'], data['ang_vel_2'], label='actual wz')
ax.set_xlabel('Timestep [ms]')
ax.set_ylabel('Tracking Error')
ax.grid(True)
ax.legend()
# ax.set_title('yaw rate tracking')

plt.tight_layout()
plt.savefig('vy_yawrate_cmd_vs_actual.jpg', dpi=300)
# plt.show()