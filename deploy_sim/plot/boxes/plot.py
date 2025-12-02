import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 场景标签（统一改这里即可） =====
RUN_TAG = "boxes"  # 例如改成 "slope10"、"flat" 等
# ===================================

# ===== 全局配置 =====
# 如果你的数据目录命名遵循 *_{RUN_TAG}_* 规则，下面这两行会自动拼接正确路径
CSV_PATHS = [
    (f"logs_csv/our_{RUN_TAG}_slim/obs.csv", "Our"),
    (f"logs_csv/dreamwaq_{RUN_TAG}/obs.csv", "DreamWaQ"),
    (f"logs_csv/baseline_{RUN_TAG}/obs.csv", "Baseline"),
]
START_STEP = 150
END_STEP = 1250
DQ_PREFIX = "dq_"
TAU_PREFIX = "tau_"

# === 平滑参数设置 ===
GLOBAL_SMOOTHING = 0.98  # 默认平滑系数
SMOOTHING_POWER = None  # 功率图平滑（None→用全局）
SMOOTHING_LINVEL = 0.98  # 线速度相关 MSE/平方 的平滑
SMOOTHING_ANGVEL = 0.98  # 角速度相关量（含 MSE）的平滑
# ====================

# 统一的输出目录（随 RUN_TAG 改变）
OUT_DIR = f"logs_csv/{RUN_TAG}"

SAVE_PATH_POWER = os.path.join(OUT_DIR, "power_compare_smooth.png")
SAVE_PATH_LINVEL0 = os.path.join(OUT_DIR, "linvel0_mse_compare_smooth.png")
SAVE_PATH_LINVEL1 = os.path.join(OUT_DIR, "linvel1_mse_compare_smooth.png")
SAVE_PATH_ANGVEL2 = os.path.join(OUT_DIR, "angvel2_mse_compare_smooth.png")
SAVE_PATH_LINVEL2SQ = os.path.join(OUT_DIR, "linvel2_sq_compare_smooth.png")
# 拆分 XY 为两个单独角速度的 MSE（相对 0）
SAVE_PATH_ANGVEL0 = os.path.join(OUT_DIR, "angvel0_mse_compare_smooth.png")
SAVE_PATH_ANGVEL1 = os.path.join(OUT_DIR, "angvel1_mse_compare_smooth.png")


# =====================


def smooth_ema(x, alpha):
    """指数滑动平均（TensorBoard 风格 EMA）"""
    if alpha <= 0:
        return np.asarray(x, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if alpha >= 1:
        return np.full_like(x, x[0])
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def compute_power_full(csv_path, alpha):
    """整段计算每步总功率，并平滑。"""
    df = pd.read_csv(csv_path)
    dq_cols = sorted([c for c in df.columns if c.startswith(DQ_PREFIX)],
                     key=lambda x: int(x[len(DQ_PREFIX):]) if x[len(DQ_PREFIX):].isdigit() else x)
    tau_cols = sorted([c for c in df.columns if c.startswith(TAU_PREFIX)],
                      key=lambda x: int(x[len(TAU_PREFIX):]) if x[len(TAU_PREFIX):].isdigit() else x)
    if "step" not in df.columns or len(dq_cols) == 0 or len(tau_cols) == 0 or len(dq_cols) != len(tau_cols):
        raise ValueError(f"{csv_path} 功率相关列缺失或数量不匹配")
    dq = np.abs(df[dq_cols].to_numpy(dtype=float))
    tau = np.abs(df[tau_cols].to_numpy(dtype=float))
    power = np.sum(dq * tau, axis=1)
    steps = df["step"].to_numpy()
    return steps, smooth_ema(power, alpha)


def compute_linvel0_mse_full(csv_path, alpha):
    df = pd.read_csv(csv_path)
    err = (df["lin_vel_0"].to_numpy(dtype=float) - df["cmd_0"].to_numpy(dtype=float)) ** 2
    steps = df["step"].to_numpy()
    return steps, smooth_ema(err, alpha)


def compute_linvel1_mse_full(csv_path, alpha):
    df = pd.read_csv(csv_path)
    err = (df["lin_vel_1"].to_numpy(dtype=float) - df["cmd_1"].to_numpy(dtype=float)) ** 2
    steps = df["step"].to_numpy()
    return steps, smooth_ema(err, alpha)


def compute_angvel2_mse_full(csv_path, alpha):
    df = pd.read_csv(csv_path)
    err = (df["ang_vel_2"].to_numpy(dtype=float) - df["cmd_2"].to_numpy(dtype=float)) ** 2
    steps = df["step"].to_numpy()
    return steps, smooth_ema(err, alpha)


def compute_linvel2_sq_full(csv_path, alpha):
    df = pd.read_csv(csv_path)
    lin2 = df["lin_vel_2"].to_numpy(dtype=float)
    sq = lin2 ** 2
    steps = df["step"].to_numpy()
    return steps, smooth_ema(sq, alpha)


def compute_angvel0_mse_full(csv_path, alpha):
    df = pd.read_csv(csv_path)
    err = (df["ang_vel_0"].to_numpy(dtype=float)) ** 2  # vs 0
    steps = df["step"].to_numpy()
    return steps, smooth_ema(err, alpha)


def compute_angvel1_mse_full(csv_path, alpha):
    df = pd.read_csv(csv_path)
    err = (df["ang_vel_1"].to_numpy(dtype=float)) ** 2  # vs 0
    steps = df["step"].to_numpy()
    return steps, smooth_ema(err, alpha)


def plot_series(series, ylabel, title, save_path, alpha):
    """通用绘图：裁剪、按 step 交集对齐后绘制"""
    if len(series) < 2:
        print("[Warn] 有数据集缺失，无法绘制。")
        return

    clipped = []
    for label, steps, values in series:
        mask = (steps >= START_STEP) & (steps < END_STEP)
        steps_clip = steps[mask]
        vals_clip = values[mask]
        clipped.append((label, steps_clip, vals_clip))

    common_steps = None
    for _, steps_clip, _ in clipped:
        s = set(steps_clip.tolist())
        common_steps = s if common_steps is None else (common_steps & s)
    if not common_steps:
        print(f"[Warn] 在区间 [{START_STEP}, {END_STEP}) 内没有重叠步。")
        return
    common_steps = np.array(sorted(common_steps), dtype=int)

    plt.figure()
    for label, steps_clip, vals_clip in clipped:
        mp = dict(zip(steps_clip.tolist(), vals_clip.tolist()))
        aligned_vals = np.array([mp[st] for st in common_steps], dtype=float)
        plt.plot(common_steps, aligned_vals, linewidth=1.6, label=label)

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(f"{title} (smoothing={alpha})")
    plt.grid(True, linestyle="-", linewidth=0.5)  # 实线网格
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"[Done] 图像已保存到 {save_path}")


def main():
    # 获取每图平滑值（若未定义则用全局默认）
    a_power = SMOOTHING_POWER if SMOOTHING_POWER is not None else GLOBAL_SMOOTHING
    a_linvel = SMOOTHING_LINVEL if SMOOTHING_LINVEL is not None else GLOBAL_SMOOTHING
    a_angvel = SMOOTHING_ANGVEL if SMOOTHING_ANGVEL is not None else GLOBAL_SMOOTHING

    # 图1: 功率
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_power_full(path, a_power)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "Total Power (|dq|·|tau|)", "Per-Step Power Comparison", SAVE_PATH_POWER, a_power)

    # 图2: lin_vel_0 MSE
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_linvel0_mse_full(path, a_linvel)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "MSE of lin_vel_0 vs cmd_0", "Per-Step MSE Comparison", SAVE_PATH_LINVEL0, a_linvel)

    # 图3: ang_vel_0 MSE（相对 0）
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_angvel0_mse_full(path, a_angvel)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "MSE of ang_vel_0 vs 0", "Per-Step MSE Comparison (ang_vel_0)", SAVE_PATH_ANGVEL0, a_angvel)

    # 图4: ang_vel_1 MSE（相对 0）
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_angvel1_mse_full(path, a_angvel)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "MSE of ang_vel_1 vs 0", "Per-Step MSE Comparison (ang_vel_1)", SAVE_PATH_ANGVEL1, a_angvel)

    # 图5: lin_vel_1 MSE
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_linvel1_mse_full(path, a_linvel)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "MSE of lin_vel_1 vs cmd_1", "Per-Step MSE Comparison (lin_vel_1)", SAVE_PATH_LINVEL1, a_linvel)

    # 图6: ang_vel_2 MSE
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_angvel2_mse_full(path, a_angvel)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "MSE of ang_vel_2 vs cmd_2", "Per-Step MSE Comparison (ang_vel_2)", SAVE_PATH_ANGVEL2, a_angvel)

    # 图7: lin_vel_2 的平方
    series = []
    for path, label in CSV_PATHS:
        try:
            steps, values = compute_linvel2_sq_full(path, a_linvel)
            series.append((label, steps, values))
        except Exception as e:
            print(f"[Warn] {path}: {e}")
    plot_series(series, "lin_vel_2²", "Per-Step lin_vel_2² Comparison", SAVE_PATH_LINVEL2SQ, a_linvel)


if __name__ == "__main__":
    main()
