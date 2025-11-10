import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 全局配置（按你的现有设置） =====
CSV_PATHS = [
    ("logs_csv/our_stair_slim/obs.csv", "Our"),
    # ("logs_csv/our_wo_contact_stair_slim/obs.csv", "Our w/o contact"),
    # ("logs_csv/our_wo_fusion_stair_slim/obs.csv", "w/o fusion"),
    ("logs_csv/dreamwaq_stair/obs.csv", "DreamWaQ"),
]
START_STEP = 1000
END_STEP = 1500
DQ_PREFIX = "dq_"
TAU_PREFIX = "tau_"

# 输出路径
SAVE_PATH_HEATMAP = "logs_csv/stair/metrics_heatmap_7x2.png"
SAVE_PATH_VALUES = "logs_csv/stair/metrics_values_7x2.csv"
SAVE_PATH_SCORES = "logs_csv/stair/metrics_scores_7x2.csv"

# ===== 指标定义（均为“越小越好”）=====
METRIC_ROWS = [
    ("lin0_mse", "lin_vel_0 MSE↓"),
    ("lin1_mse", "lin_vel_1 MSE↓"),
    ("ang2_mse", "ang_vel_2 MSE↓"),
    ("ang0_sq", "ang_vel_0²↓"),
    ("ang1_sq", "ang_vel_1²↓"),
    ("lin2_sq", "lin_vel_2²↓"),
    ("power_mean", "power (avg)↓"),
]


# ===== 工具函数 =====
def _sorted_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"找不到列名前缀: {prefix}")
    try:
        cols = sorted(cols, key=lambda x: int(x[len(prefix):]))
    except Exception:
        cols = sorted(cols)  # 回退
    return cols


def _require_cols(df, needed, path):
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺少列: {miss}")


def compute_power_mean(df):
    dq_cols = _sorted_cols(df, DQ_PREFIX)
    tau_cols = _sorted_cols(df, TAU_PREFIX)
    if len(dq_cols) != len(tau_cols):
        raise ValueError(f"列数量不匹配: dq={len(dq_cols)} vs tau={len(tau_cols)}")

    dq = np.abs(df[dq_cols].to_numpy(dtype=float))
    tau = np.abs(df[tau_cols].to_numpy(dtype=float))
    per_step_power = np.sum(dq * tau, axis=1)
    return float(np.mean(per_step_power))


def compute_metrics_for_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    _require_cols(df, ["step"], csv_path)

    # 只取指定步数区间
    mask = (df["step"].to_numpy() >= START_STEP) & (df["step"].to_numpy() < END_STEP)
    if not np.any(mask):
        raise ValueError(f"{csv_path} 在区间 [{START_STEP}, {END_STEP}) 内无数据")
    d = df.loc[mask].reset_index(drop=True)

    # 需要的列检查
    need_cols = [
        "lin_vel_0", "cmd_0",
        "lin_vel_1", "cmd_1",
        "ang_vel_2", "cmd_2",
        "ang_vel_0", "ang_vel_1",
        "lin_vel_2",
    ]
    _require_cols(d, need_cols, csv_path)
    # power 相关列检查在 compute_power_mean 内做

    # 逐项计算
    lin0_mse = np.mean((d["lin_vel_0"].to_numpy(float) - d["cmd_0"].to_numpy(float)) ** 2)
    lin1_mse = np.mean((d["lin_vel_1"].to_numpy(float) - d["cmd_1"].to_numpy(float)) ** 2)
    ang2_mse = np.mean((d["ang_vel_2"].to_numpy(float) - d["cmd_2"].to_numpy(float)) ** 2)

    ang0_sq = np.mean((d["ang_vel_0"].to_numpy(float)) ** 2)
    ang1_sq = np.mean((d["ang_vel_1"].to_numpy(float)) ** 2)
    lin2_sq = np.mean((d["lin_vel_2"].to_numpy(float)) ** 2)

    power_avg = compute_power_mean(d)

    return {
        "lin0_mse": lin0_mse,
        "lin1_mse": lin1_mse,
        "ang2_mse": ang2_mse,
        "ang0_sq": ang0_sq,
        "ang1_sq": ang1_sq,
        "lin2_sq": lin2_sq,
        "power_mean": power_avg,
    }


def minmax_to_score_min_better(values_row):
    """把一行的原值(越小越好)映射到score∈[0,1]，1为最好"""
    v = np.array(values_row, dtype=float)
    vmin, vmax = np.min(v), np.max(v)
    if np.isclose(vmin, vmax):
        return np.full_like(v, 0.5, dtype=float)
    return 1.0 - (v - vmin) / (vmax - vmin)


# ===== 主流程 =====
def main():
    # 计算数值表（原值）
    cols = []
    col_names = []
    for path, label in CSV_PATHS:
        stats = compute_metrics_for_csv(path)
        col = [stats[key] for key, _nice in METRIC_ROWS]  # 按既定顺序取值
        cols.append(col)
        col_names.append(label)

    vals = pd.DataFrame(
        data=np.column_stack(cols),
        index=[nice for _, nice in METRIC_ROWS],
        columns=col_names
    )

    # 归一化得分（越大越好，仅用于着色）
    scores = pd.DataFrame(index=vals.index, columns=vals.columns, dtype=float)
    for r in vals.index:
        scores.loc[r] = minmax_to_score_min_better(vals.loc[r].values)

    # 平均名次（1=最好，2=次之；这里只有两列）
    ranks = vals.rank(axis=1, ascending=True, method="average")  # 越小越好
    avg_rank = ranks.mean(axis=0).sort_values()
    print("\n=== 平均名次（Avg. Rank, 越小越好）===\n", avg_rank)

    # 相对 DreamWaQ 的改进百分比（越小越好场景： (base - val)/base ）
    if "DreamWaQ" in vals.columns:
        base = vals["DreamWaQ"]
        print("\n=== 相对 DreamWaQ 的改进(%)（正数=更好）===")
        for meth in vals.columns:
            if meth == "DreamWaQ":
                continue
            imp = (base - vals[meth]) / base * 100.0
            print(f"\n[{meth}]")
            print((imp).rename(lambda s: f"{s}").to_string(float_format=lambda x: f"{x:,.2f}"))
    else:
        print("\n[提示] 未找到 DreamWaQ 列，跳过相对改进统计。")

    # 保存原值和得分
    os.makedirs(os.path.dirname(SAVE_PATH_VALUES), exist_ok=True)
    vals.to_csv(SAVE_PATH_VALUES)
    scores.to_csv(SAVE_PATH_SCORES)
    print(f"\n[Saved] 原值表: {SAVE_PATH_VALUES}")
    print(f"[Saved] 得分表: {SAVE_PATH_SCORES}")

    # 画热力表
    fig_h, ax = plt.subplots(figsize=(4.8, 3.6))  # 两列时这个尺寸足够清晰
    im = ax.imshow(scores.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # 轴刻度
    ax.set_xticks(np.arange(len(vals.columns)))
    ax.set_xticklabels(vals.columns, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(vals.index)))
    ax.set_yticklabels(vals.index)

    # 单元格写入原值（科学计数或三位有效数字）
    def fmt(x):
        # 兼顾数量级差异
        if x == 0:
            return "0"
        ax_abs = abs(x)
        if ax_abs < 1e-3 or ax_abs >= 1e3:
            return f"{x:.2e}"
        return f"{x:.3g}"

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, fmt(float(vals.iloc[i, j])), va="center", ha="center", fontsize=8, color="black")

    cbar = fig_h.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Normalized score (higher is better)")

    ax.set_title("Seven-Metric Heatmap (lower is better for raw values)")
    ax.set_xlabel("Methods")
    ax.set_ylabel("Metrics")

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH_HEATMAP), exist_ok=True)
    plt.savefig(SAVE_PATH_HEATMAP, dpi=200)
    print(f"[Saved] 热力表图像: {SAVE_PATH_HEATMAP}")


if __name__ == "__main__":
    main()
