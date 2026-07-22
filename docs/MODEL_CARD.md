# DARNet-ETTh1 Model Card

## Model overview

DARNet-ETTh1 is an experimental multivariate time-series forecasting system for
transformer oil temperature prediction. It uses 72 hours of historical ETTh1
observations to predict the target variable `OT` over the next 24 hours.

The project adapts sequence representation and long-range dependency modeling
ideas associated with the original DARNet auditory attention detection model to
a continuous forecasting task. It evaluates Patch representation, attention,
and Fourier seasonal features through six controlled ablation modes.

## Intended use

- Reproducing the undergraduate thesis experiments archived in this repository.
- Studying component-level effects in a controlled time-series ablation setup.
- Loading the provided checkpoints for educational evaluation and visualization.
- Serving as a baseline for later multi-horizon or multi-dataset experiments.

This repository is a research prototype. It is not intended for direct use in
safety-critical power-grid operation or automated equipment control.

## Inputs and outputs

| Item | Description |
|---|---|
| Input | 72 hourly observations of 7 ETTh1 numerical variables |
| Output | 24-step forecast of transformer oil temperature `OT` |
| Dataset | ETTh1, 17,420 hourly records |
| Primary metric | RMSE |
| Additional metrics | MAE, RRSE, CORR, SMAPE and MAPE |

## Evaluated configurations

| Mode | Patch | Attention | Seasonality |
|---|:---:|:---:|:---:|
| `baseline` | No | No | No |
| `patch` | Yes | No | No |
| `attention` | No | Yes | No |
| `patch_attention` | Yes | Yes | No |
| `season` | No | No | Yes |
| `full_model` | Yes | Yes | Yes |

Under the archived unified setup, `patch` achieves the lowest RMSE of `13.0248`,
approximately `5.49%` below the baseline RMSE of `13.7816`. The full combination
does not outperform the Patch-only mode, so the repository does not claim that
every added component produces an independent improvement.

## Reproducibility

- Default random seed: `42`
- Input window: `72`
- Forecast horizon: `24`
- Batch size: `256`
- Hidden dimension: `64`
- Number of layers: `2`
- Archived checkpoints: `pretrained/`
- Archived metrics: `results/all_modes_summary.csv`
- Unified runner: `code/run_all_modes.py`
- Checkpoint evaluator: `code/evaluate_checkpoint.py`

Results can vary slightly across operating systems, GPU models, CUDA versions,
PyTorch versions, and low-level numerical libraries. The files under `results/`
are the reference values used by this repository.

## Limitations

- Evaluation currently focuses on one dataset and one 72-to-24 forecasting task.
- The archived experiment does not include repeated-seed confidence intervals.
- The full combination of modules has not yet been exhaustively tuned.
- The project does not currently compare all methods under one external benchmark.
- Percentage metrics can be unstable when target values are close to zero; RMSE is
  therefore treated as the primary comparison metric.

## Data and licensing

The unmodified `ETTh1.csv` file originates from
[zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) and remains subject
to its CC BY-ND 4.0 license. Project-authored code and documentation use the MIT
License. See `THIRD_PARTY_NOTICES.md` for details.
