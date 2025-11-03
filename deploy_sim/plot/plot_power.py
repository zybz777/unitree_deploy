import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 全局配置 =====
CSV_PATHS = [
    ("logs_csv/our_flat/obs.csv",       "Our"),
    ("logs_csv/dreamwaq_flat/obs.csv",  "DreamWaQ"),
]
START_STEP = 100
END_STEP   = 1000
DQ_PREFIX  = "dq_"
TAU_PREFIX = "tau_"
SMOOTHING  = 0.95        # 0.0 = 不平滑, 1.0 = 非常平滑
SAVE_PATH  = "logs_csv/power_compare_smooth.png"
# =====================

def smooth_ema(x, alpha):
    """指数滑动平均（TensorBoard 风格 EMA）在全序列上进行。"""
    if alpha <= 0:
        return x.copy()
    if alpha >= 1:
        return np.full_like(x, x[0])
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y

def compute_power_full(csv_path):
    """用整份 CSV 计算每步总功率，并在整段上做平滑；返回 steps, power_smooth（全长）。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 'step' 列")

    dq_cols  = sorted([c for c in df.columns if c.startswith(DQ_PREFIX)],
                      key=lambda x: int(x[len(DQ_PREFIX):]))
    tau_cols = sorted([c for c in df.columns if c.startswith(TAU_PREFIX)],
                      key=lambda x: int(x[len(TAU_PREFIX):]))
    if len(dq_cols) == 0 or len(tau_cols) == 0:
        raise ValueError(f"{csv_path} 找不到列名前缀: dq={DQ_PREFIX}, tau={TAU_PREFIX}")
    if len(dq_cols) != len(tau_cols):
        raise ValueError(f"{csv_path} 列数量不匹配: dq={len(dq_cols)} vs tau={len(tau_cols)}")

    # 整段功率（正值定义：|dq|*|tau|）
    dq   = np.abs(df[dq_cols].to_numpy(dtype=float))
    tau  = np.abs(df[tau_cols].to_numpy(dtype=float))
    power = np.sum(dq * tau, axis=1)          # 全长度
    steps = df["step"].to_numpy()

    # 在整段上做平滑，然后再裁剪显示范围
    power_smooth = smooth_ema(power, SMOOTHING)
    return steps, power_smooth

def main():
    plt.figure()
    for path, label in CSV_PATHS:
        try:
            steps_all, power_sm_all = compute_power_full(path)
        except Exception as e:
            print(f"[Warn] {path}: {e}")
            continue

        # 仅裁剪显示范围（不影响平滑本身）
        mask = (steps_all >= START_STEP) & (steps_all < END_STEP)
        if not np.any(mask):
            print(f"[Warn] {path} 在区间 [{START_STEP}, {END_STEP}) 内无数据，跳过。")
            continue

        plt.plot(steps_all[mask], power_sm_all[mask], linewidth=1.6, label=label)

    plt.xlabel("Step")
    plt.ylabel("Total Power (|dq|·|tau|)")
    plt.title(f"Per-Step Power Comparison (smoothing={SMOOTHING})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=200)
    print(f"[Done] 图像已保存到 {SAVE_PATH}")

if __name__ == "__main__":
    main()
