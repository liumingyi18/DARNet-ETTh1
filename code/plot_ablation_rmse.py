from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


MODE_ORDER = ["baseline", "patch", "attention", "patch_attention", "season", "full_model"]
DISPLAY_LABELS = {
    "baseline": "Baseline",
    "patch": "Patch",
    "attention": "Attention",
    "patch_attention": "Patch+Attention",
    "season": "Season",
    "full_model": "Full Model",
}
OUTPUT_NAME = "darnet_ablation_rmse_comparison.png"


def get_available_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "KaiTi",
        "FangSong",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            return font_name
    return None


def resolve_csv_path() -> Path:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"找不到指定的 CSV 文件：{path}")
        return path

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    cwd = Path.cwd()

    candidates = [
        script_dir / "all_modes_summary.csv",
        cwd / "all_modes_summary.csv",
        project_root / "results" / "all_modes_summary.csv",
        cwd / "results" / "all_modes_summary.csv",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError("未找到 all_modes_summary.csv，请将其放在脚本同目录下，或通过命令行传入文件路径。")


def main():
    csv_path = resolve_csv_path()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME

    font_name = get_available_chinese_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(csv_path)

    if "mode" not in df.columns or "RMSE" not in df.columns:
        raise ValueError("CSV 文件中必须包含 'mode' 和 'RMSE' 两列。")

    df = df[df["mode"].isin(MODE_ORDER)].copy()
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    df = df.sort_values("mode")

    missing_modes = [m for m in MODE_ORDER if m not in df["mode"].astype(str).tolist()]
    if missing_modes:
        raise ValueError(f"CSV 中缺少以下 mode 数据：{missing_modes}")

    modes = [DISPLAY_LABELS[mode] for mode in df["mode"].astype(str).tolist()]
    rmse_values = df["RMSE"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    bars = ax.bar(modes, rmse_values, color=colors, width=0.65, edgecolor="black", linewidth=0.8)

    ax.set_title("DARNet 消融实验：不同模式 RMSE 对比", fontsize=16, pad=14)
    ax.set_xlabel("模式", fontsize=13)
    ax.set_ylabel("RMSE", fontsize=13)

    y_max = max(rmse_values)
    y_range = max(y_max, 1e-6)
    ax.set_ylim(0, y_max + y_range * 0.08)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, rmse_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_range * 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"图片已保存到：{output_path}")


if __name__ == "__main__":
    main()
