from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


DATA_PATH = Path("./data/ETTh1.csv")
FIGURE_DIR = Path("./figures")
FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
FEATURE_LABELS = ["高压负荷下界", "高压负荷上界", "中压负荷下界", "中压负荷上界", "低压负荷下界", "低压负荷上界", "变压器油温"]


def setup_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "SimSun", "NSimSun", "KaiTi", "FangSong"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    available = [font for font in candidates if font in installed]
    if available:
        plt.rcParams["font.sans-serif"] = available
    else:
        plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False


def load_etth1_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件：{DATA_PATH.resolve()}")

    data = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"数据文件缺少字段：{missing_columns}")
    return data[FEATURE_COLUMNS].copy()


def draw_ot_trend(data):
    ot = data["OT"].dropna()
    rolling_mean = ot.rolling(window=24, min_periods=1).mean()
    global_mean = float(ot.mean())
    global_std = float(ot.std())

    zoom_start = pd.Timestamp("2016-07-01")
    zoom_end = pd.Timestamp("2016-07-08")
    zoom_mask = (ot.index >= zoom_start) & (ot.index <= zoom_end)

    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(ot.index, ot.values, color="#4C72B0", linewidth=0.6, alpha=0.55, label="OT小时值")
    ax.plot(rolling_mean.index, rolling_mean.values, color="#D55E00", linewidth=1.25, label="24小时滑动均值")
    ax.axhline(global_mean, color="#333333", linestyle="--", linewidth=1.1, label="均值")
    ax.axvspan(zoom_start, zoom_end, color="#7BC96F", alpha=0.18)

    stats_text = (
        f"样本数：{len(ot)}\n"
        f"时间范围：{ot.index.min():%Y-%m-%d} 至 {ot.index.max():%Y-%m-%d}\n"
        f"均值：{global_mean:.2f}，标准差：{global_std:.2f}"
    )
    ax.text(
        0.02,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
    )

    shade_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
    ax.annotate(
        "局部放大区间",
        xy=(zoom_start + (zoom_end - zoom_start) / 2, shade_y),
        xytext=(pd.Timestamp("2016-09-01"), shade_y + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.18),
        arrowprops={"arrowstyle": "->", "color": "#4B8B3B", "linewidth": 0.8},
        color="#2F6B2F",
        fontsize=10.5,
        ha="center",
    )

    axins = inset_axes(ax, width="42%", height="36%", loc="upper right", borderpad=1.1)
    axins.set_facecolor("white")
    axins.plot(ot.index[zoom_mask], ot.values[zoom_mask], color="#4C72B0", linewidth=0.75, alpha=0.7)
    axins.plot(
        rolling_mean.index[zoom_mask],
        rolling_mean.values[zoom_mask],
        color="#D55E00",
        linewidth=1.1,
    )
    axins.axhline(global_mean, color="#333333", linestyle="--", linewidth=0.9)
    axins.text(
        0.04,
        0.92,
        "局部放大：2016-07-01 至 2016-07-08",
        transform=axins.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.85},
    )
    axins.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axins.tick_params(axis="x", labelsize=8, rotation=30)
    axins.tick_params(axis="y", labelsize=8)
    axins.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.25)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#888888", linewidth=0.7)

    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("OT值", fontsize=12)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=10)
    ax.grid(axis="both", linestyle="--", linewidth=0.55, alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=25)

    fig.tight_layout()
    output_path = FIGURE_DIR / "etth1_ot_time_series.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def draw_correlation_heatmap(data):
    corr = data[FEATURE_COLUMNS].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.2, 7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("相关系数", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_yticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_LABELS, fontsize=9.5)
    ax.set_yticklabels(FEATURE_LABELS, fontsize=9.5)

    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.6 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9.5)

    ax.set_xticks(np.arange(-0.5, len(FEATURE_COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FEATURE_COLUMNS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", top=False, bottom=True, labelrotation=30)

    fig.tight_layout()
    output_path = FIGURE_DIR / "etth1_feature_correlation_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    setup_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_etth1_data()

    ot_figure_path = draw_ot_trend(data)
    corr_figure_path = draw_correlation_heatmap(data)

    print(f"OT时间序列趋势图已保存：{ot_figure_path.resolve()}")
    print(f"多变量特征相关性热力图已保存：{corr_figure_path.resolve()}")


if __name__ == "__main__":
    main()
