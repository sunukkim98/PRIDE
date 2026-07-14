# PRIDE

[TODO: 논문 제목]

This repository is the official implementation of **PRIDE**, a denoising method for
implicit-feedback recommender systems.

> **Note:** This codebase was originally built on top of the [PLD (WWW 2025)](https://arxiv.org/pdf/2502.00348)
> implementation and has since been extended with PRIDE and several baselines
> (BOD, DCF, T-CE, R-CE, REQUIEM). [TODO: 원 저자 attribution / 라이선스 문구 확인 후 정리]

## Authors
[TODO: 저자 목록]

## Abstract
[TODO: abstract]

## Environment

```bash
conda env create -f environment.yml
conda activate requiem
```

or, with pip:

```bash
pip install -r requirements.txt
```

- python >= 3.8
- torch >= 1.10.1
- numpy >= 1.22.2
- scikit-learn >= 1.0.2
- scipy >= 1.8.0

## Datasets

Place preprocessed data under `data/<Dataset>/`. Evaluated datasets:
Amazon-Book, MIND, Yelp.

## Usage (Quick Start)

Run PRIDE with an MF or LightGCN backbone on a given dataset:

```bash
python main.py --model MF --dataset MIND --method PRIDE
```

Key hyperparameters (see `meta_config.py` for the full list):

| Flag | Description |
| --- | --- |
| `--model` | backbone: `MF`, `LightGCN`, `NeuMF` |
| `--dataset` | dataset name under `data/` |
| `--method` | denoising method: `PRIDE`, `Origin`, `PLD`, `REQUIEM`, `BOD`, `DCF`, `TCE`, `RCE`, ... |
| `--noise` | synthetic noise ratio injected into training interactions |
| `--begin_adv`, `--ema`, `--num_codebook`, `--energy_r`, `--energy_lambda` | PRIDE-specific hyperparameters |

A single-run example with PRIDE's best hyperparameters is in
[`scripts/run_single.sh`](scripts/run_single.sh) / [`scripts/run_single.slurm`](scripts/run_single.slurm).

### Reproducing paper results

[TODO: 논문 Table/Figure ↔ scripts/config 매핑. 예: "Table 2 (MIND, MF) = scripts/run_single_mf_mind.slurm"]

## Repository Structure

```
main.py                 # entry point (python main.py --model ... --dataset ... --method ...)
meta_config.py           # CLI argument definitions
utls/trainer.py          # training loops for each method (PRIDE + baselines)
utls/model_config.py     # per-method hyperparameter config builders
model/                    # backbone models (MF, LightGCN, NeuMF)
vector_quantize_pytorch/ # vector-quantization codebook (used by PRIDE)
config/                  # W&B sweep configs for experiments
scripts/                 # Slurm submission scripts
analyze/                 # result aggregation / plotting scripts
```

## Citation

If you find this work useful, please cite:

```bibtex
[TODO: BibTeX]
```
