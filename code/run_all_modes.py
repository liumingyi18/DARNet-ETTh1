"""Run DARNet ablation modes without overwriting the archived reference results."""

import argparse
import importlib.util
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


MODE_CONFIGS = {
    "baseline": {"patch_size": 1, "use_attention": False, "use_seasonality": False},
    "patch": {"patch_size": 4, "use_attention": False, "use_seasonality": False},
    "attention": {"patch_size": 1, "use_attention": True, "use_seasonality": False},
    "patch_attention": {"patch_size": 4, "use_attention": True, "use_seasonality": False},
    "season": {"patch_size": 1, "use_attention": False, "use_seasonality": True},
    "full_model": {"patch_size": 4, "use_attention": True, "use_seasonality": True},
}


def load_training_module(project_root):
    module_path = project_root / "code" / "0114-DARNet-V4-opt.py"
    spec = importlib.util.spec_from_file_location("darnet_opt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载训练代码：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="运行 DARNet 六种消融模式并生成独立汇总结果")
    parser.add_argument(
        "--mode",
        choices=[*MODE_CONFIGS, "all"],
        default="all",
        help="要运行的模式，默认依次运行全部六种模式",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录；不指定时自动写入 outputs/run_时间_seed随机种子",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    module = load_training_module(project_root)

    if args.output:
        output_root = Path(args.output)
        if not output_root.is_absolute():
            output_root = project_root / output_root
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = project_root / "outputs" / f"run_{timestamp}_seed{args.seed}"

    result_dir = output_root / "results"
    checkpoint_dir = output_root / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    modes = list(MODE_CONFIGS) if args.mode == "all" else [args.mode]
    summary_rows = []

    for mode in modes:
        config = MODE_CONFIGS[mode]
        result_path = result_dir / f"{mode}.csv"
        checkpoint_path = checkpoint_dir / f"{mode}.pt"
        print(f"\n===== 开始运行模式：{mode} =====")

        try:
            module.run_training(
                patch_size=config["patch_size"],
                use_attention=config["use_attention"],
                use_seasonality=config["use_seasonality"],
                save_path=str(checkpoint_path),
                results_path=str(result_path),
                seed=args.seed,
            )
            row = pd.read_csv(result_path).iloc[0].to_dict()
            row.update(
                {
                    "mode": mode,
                    "use_seasonality": config["use_seasonality"],
                    "seed": args.seed,
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            row = {
                "mode": mode,
                "patch_size": config["patch_size"],
                "use_attention": config["use_attention"],
                "use_seasonality": config["use_seasonality"],
                "seed": args.seed,
                "status": "failed",
                "error": repr(exc),
            }
            print(f"模式 {mode} 运行失败：{exc}")

        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            result_dir / "all_modes_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n===== 运行结束 =====")
    print(f"结果目录：{result_dir.resolve()}")
    print(f"权重目录：{checkpoint_dir.resolve()}")
    print(f"汇总文件：{(result_dir / 'all_modes_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
