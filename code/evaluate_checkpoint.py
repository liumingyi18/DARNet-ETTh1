"""Evaluate an archived DARNet state_dict without retraining."""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


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
        raise ImportError(f"无法加载模型代码：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(module, model, test_x, test_y, scaler_y, patch_size, target_index, batch_size):
    if patch_size > 1:
        test_x, test_y = module.slice_data(test_x, test_y, patch_size)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(module.device)
            labels = labels.to(module.device)

            if patch_size > 1:
                decoder_step = module.extract_last_timestep_from_patched_inputs(
                    inputs, labels.shape[2]
                ).clone()
            else:
                decoder_step = inputs[:, -1:, :].clone()

            predicted_steps = []
            for step in range(labels.shape[1]):
                outputs = model(inputs, decoder_step, target_index=target_index)
                if outputs.dim() == 2:
                    current_prediction = outputs[:, -1:].unsqueeze(1)
                else:
                    current_prediction = outputs[:, -1:, 0:1]
                predicted_steps.append(current_prediction.squeeze(-1))

                if step < labels.shape[1] - 1:
                    next_step = labels[:, step : step + 1, :].clone()
                    next_step[:, :, target_index : target_index + 1] = current_prediction
                    decoder_step = torch.cat((decoder_step, next_step), dim=1)

            predictions.append(torch.cat(predicted_steps, dim=1).cpu().numpy())
            targets.append(labels[:, :, target_index].cpu().numpy())

    prediction = np.concatenate(predictions).reshape(-1, 1)
    target = np.concatenate(targets).reshape(-1, 1)
    prediction = scaler_y.inverse_transform(prediction)
    target = scaler_y.inverse_transform(target)

    mean_prediction = np.mean(prediction)
    mean_target = np.mean(target)
    mape = np.mean(np.abs(target - prediction) / (np.abs(target) + 1e-8))
    smape = 2 * np.mean(
        np.abs(target - prediction) / (np.abs(target) + np.abs(prediction) + 1e-8)
    )
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean(np.square(target - prediction)))
    rrse = np.sqrt(np.sum(np.square(target - prediction))) / (
        np.sqrt(np.sum(np.square(target - mean_target))) + 1e-8
    )
    corr = np.sum((target - mean_target) * (prediction - mean_prediction)) / (
        np.sqrt(np.sum((target - mean_target) ** 2))
        * np.sqrt(np.sum((prediction - mean_prediction) ** 2))
        + 1e-8
    )
    return {
        "MAPE": float(mape),
        "SMAPE": float(smape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "RRSE": float(rrse),
        "CORR": float(corr),
    }


def main():
    parser = argparse.ArgumentParser(description="评估归档中的 DARNet 预训练权重")
    parser.add_argument("mode", choices=MODE_CONFIGS, help="要评估的模式")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=str, default=None, help="可选的结果 CSV 路径")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    module = load_training_module(project_root)
    config = MODE_CONFIGS[args.mode]

    module.random_seed_set(42)
    module.ENABLE_PATCH = config["patch_size"] > 1
    module.PATCH_SIZE = config["patch_size"]

    data = module.load_etth1_data(use_seasonality=config["use_seasonality"])
    target_index = data.columns.get_loc("OT")
    split_index = int(len(data) * 0.8)
    train_valid_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    _, scaler, scaler_y = module.normalization(train_valid_data)
    normalized_test_data = scaler.transform(test_data)
    test_x, test_y = module.series_to_supervise(normalized_test_data, 72, 24)

    model = module.DARNet(
        train_valid_data.shape[1],
        [64, 64],
        72,
        [64, 64],
        dropout=0,
        use_attention=config["use_attention"],
    ).to(module.device)

    checkpoint_path = project_root / "pretrained" / f"{args.mode}.pt"
    state_dict = torch.load(checkpoint_path, map_location=module.device)
    model.load_state_dict(state_dict)

    metrics = evaluate(
        module,
        model,
        test_x,
        test_y,
        scaler_y,
        config["patch_size"],
        target_index,
        args.batch_size,
    )
    row = {"mode": args.mode, **config, **metrics}
    print(pd.DataFrame([row]).to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"评估结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()
