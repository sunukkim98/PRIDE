# PRIDE
This repository provides the official implementation of **PRIDE**.
* **PRIDE: Preference-structure and Representation-stability based Interaction Denoising** </br>
Sunuk Kim<sup>\*</sup>, Minseo Jeon<sup>\*</sup>, Daewon Gwak, Gyuwon Je, and Jinhong Jung<sup>†</sup> </br>
Soongsil University, Seoul, South Korea </br>
<sup>\*</sup>Equal contribution, <sup>†</sup>Corresponding author </br>
Venue: T.B.D

## 📝 Abstract
Implicit feedback serves as the primary training signal for recommender systems due to its ease
of collection, yet it contains noisy interactions that do not reflect users' true preferences.
Effective denoising is therefore essential for reliable preference learning. However, existing
denoising methods evaluate each interaction in isolation from the user's broader preference
structure, and rely on signals observed at a fixed point of training, thereby failing to capture
representation dynamics across training. To address these limitations, we propose **PRIDE**
(**P**reference-structure and **R**epresentation-stability based **I**nteraction **D**enoising), a
novel denoising framework for implicit feedback recommendation. PRIDE estimates the reliability of
each interaction from two complementary signals, consistency with the user's preference structure
and representation stability across training, and combines them into a per-interaction weight
applied to the BPR loss. Extensive experiments across multiple datasets and noise settings
demonstrate that PRIDE achieves stable performance and robust representation learning compared
with existing denoising methods.

## ⚙️ Prerequisites
You should install the required packages with a conda environment by typing the following command in your terminal:
```bash
conda env create -f environment.yml
conda activate pride
```

All experiments in the paper were run on a single NVIDIA RTX 4090 GPU (24GB VRAM).

> **Note:** This codebase was originally built on top of the [PLD (WWW 2025)](https://github.com/Kaike-Zhang/PLD)
> implementation and has since been extended with PRIDE and several baselines
> (T-CE, R-CE, DCF, BOD). We thank the PLD authors for making their code available.

## 📊 Datasets
We evaluate PRIDE on three public benchmark datasets that cover diverse domains. Yelp2018 is a
dataset of user reviews on businesses; MIND is a news recommendation dataset based on user click
records; Amazon-Book is a dataset of user purchases/reviews from the Book category of Amazon.

| Dataset | # Users | # Items | # Interactions | Avg. Length | Sparsity (%) |
|:--|--:|--:|--:|--:|--:|
| Yelp2018 | 31,668 | 38,048 | 1,561,406 | 49.3 | 99.88 |
| MIND | 38,441 | 38,000 | 1,210,953 | 31.5 | 99.92 |
| Amazon-Book | 52,643 | 91,599 | 2,704,860 | 51.3 | 99.95 |

For Yelp2018 and MIND we use the preprocessed versions provided by
[PLD](https://github.com/Kaike-Zhang/PLD), and for Amazon-Book, the version provided by
[LightGCN++](https://github.com/geon0325/LightGCNpp).

Place the preprocessed data under `data/<Dataset>/` (e.g. `data/MIND/data.json`) before running
the code; a `data/<Dataset>/processed/` cache is generated automatically on first run.

## 🚀 Usage of PRIDE
PRIDE trains in two stages — warm-up initialization followed by reweighted training — but both run
within a single command; the best checkpoint (by validation `Recall@20`) is selected automatically
and evaluated on the test split at the end of training.

```bash
python main.py --model MF --dataset MIND --method PRIDE
```

- `--model`: backbone used in the paper's experiments is `MF` or `LightGCN` (`NeuMF` is also
  supported by the codebase but was not evaluated in the paper).
- `--method`: `PRIDE`, or one of the baselines compared in the paper — `Origin` (no denoising),
  `PLD`, `BOD` (learning-based), `TCE`/`RCE`/`DCF` (loss-based, T-CE/R-CE/DCF in the paper). See
  `meta_config.py` for the full argument list.

**Note:** `meta_config.py`'s defaults are the validated hyperparameters for **MF on MIND**, so the
command above reproduces that result as-is. For the other five (backbone, dataset) combinations,
use the matching script under `scripts/` (e.g. `scripts/run_pride_mf_yelp.sh`,
`scripts/run_pride_lgn_amazon_book.sh`, ...) or pass the hyperparameters explicitly — see
`config/pride_<backbone>_<dataset>.yaml` for the validated values of each combination.

## 📈 Experimental Results of `PRIDE`

### Performance for top-*k* item recommendation
The reported results in the paper are as follows (Table 3; best in **bold**, second-best
underlined).

**MF backbone**

| Model | Yelp2018 R@20 | Yelp2018 R@50 | Yelp2018 N@20 | Yelp2018 N@50 | MIND R@20 | MIND R@50 | MIND N@20 | MIND N@50 | Amazon-Book R@20 | Amazon-Book R@50 | Amazon-Book N@20 | Amazon-Book N@50 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MF (Origin) | 0.0689 | 0.1310 | 0.0531 | 0.0773 | 0.0675 | 0.1252 | 0.0439 | 0.0628 | 0.0669 | 0.1200 | 0.0500 | 0.0700 |
| +R-CE | 0.0722 | 0.1376 | 0.0557 | 0.0811 | 0.0734 | 0.1354 | 0.0479 | 0.0682 | 0.0669 | 0.1210 | 0.0501 | 0.0705 |
| +T-CE | 0.0557 | 0.1103 | 0.0414 | 0.0624 | 0.0568 | 0.1098 | 0.0360 | 0.0533 | 0.0524 | 0.0985 | 0.0388 | 0.0559 |
| +BOD | 0.0580 | 0.1119 | 0.0448 | 0.0655 | 0.0648 | 0.1190 | 0.0430 | 0.0605 | 0.0466 | 0.0860 | 0.0345 | 0.0493 |
| +DCF | 0.0658 | 0.1254 | 0.0509 | 0.0742 | 0.0694 | 0.1271 | 0.0455 | 0.0645 | 0.0556 | 0.1026 | 0.0422 | 0.0601 |
| +PLD | _0.0723_ | _0.1385_ | _0.0559_ | _0.0816_ | _0.0771_ | _0.1393_ | _0.0517_ | _0.0720_ | _0.0675_ | _0.1214_ | _0.0504_ | _0.0708_ |
| **+PRIDE** | **0.0760** | **0.1434** | **0.0590** | **0.0852** | **0.0808** | **0.1444** | **0.0544** | **0.0752** | **0.0694** | **0.1247** | **0.0527** | **0.0735** |

Δ over best baseline: +2.8% – +5.7% (Recall/NDCG). Δ over MF (Origin): +3.7% – +24.0%.

**LightGCN backbone**

| Model | Yelp2018 R@20 | Yelp2018 R@50 | Yelp2018 N@20 | Yelp2018 N@50 | MIND R@20 | MIND R@50 | MIND N@20 | MIND N@50 | Amazon-Book R@20 | Amazon-Book R@50 | Amazon-Book N@20 | Amazon-Book N@50 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| LightGCN (Origin) | **0.0862** | 0.1563 | **0.0680** | **0.0953** | 0.0930 | 0.1631 | _0.0628_ | _0.0859_ | 0.0775 | 0.1368 | _0.0590_ | _0.0814_ |
| +R-CE | 0.0849 | 0.1559 | _0.0668_ | 0.0944 | 0.0936 | 0.1645 | 0.0620 | 0.0854 | 0.0750 | 0.1347 | 0.0567 | 0.0792 |
| +T-CE | 0.0714 | 0.1337 | 0.0558 | 0.0799 | 0.0663 | 0.1214 | 0.0440 | 0.0618 | 0.0472 | 0.0894 | 0.0354 | 0.0512 |
| +BOD | 0.0629 | 0.1204 | 0.0489 | 0.0712 | 0.0722 | 0.1299 | 0.0464 | 0.0653 | 0.0510 | 0.0961 | 0.0381 | 0.0552 |
| +DCF | 0.0799 | 0.1496 | 0.0629 | 0.0900 | 0.0871 | 0.1566 | 0.0582 | 0.0810 | 0.0678 | 0.1232 | 0.0513 | 0.0723 |
| +PLD | 0.0844 | 0.1552 | _0.0668_ | 0.0944 | 0.0662 | 0.1214 | 0.0439 | 0.0618 | 0.0767 | 0.1366 | 0.0587 | 0.0813 |
| **+PRIDE** | _0.0851_ | **0.1572** | 0.0665 | 0.0945 | **0.0950** | **0.1681** | **0.0630** | **0.0870** | **0.0791** | **0.1389** | **0.0600** | **0.0827** |

PRIDE ranks first on every metric with MF, and on all metrics on MIND and Amazon-Book with
LightGCN; on Yelp2018 + LightGCN the gap to the origin backbone is marginal (LightGCN's graph
smoothing already provides some implicit denoising there). PRIDE consistently outperforms PLD, the
strongest baseline, on every dataset and metric with both backbones.

### Noise robustness
Under synthetic noise injection (0%–40%) on Yelp2018 and MIND with both backbones, PRIDE
maintains the highest Recall@20 and degrades the least as the noise ratio increases (see Figure 2
in the paper).

### Validated hyperparameters
Fixed across all settings: embedding dimension `d=64`, batch size `2048`, up to 100 epochs with
AdamW. The remaining hyperparameters were tuned via grid search on the validation set:

| Hyperparameter | Search space |
|---|---|
| Learning rate `--lr` | {1e-1, 1e-2, 1e-3, 1e-4, 1e-5} |
| Weight decay `--weight_decay` | {1e-1, 1e-2, 1e-3, 1e-4, 1e-5} |
| Warm-up length `--begin_adv` | {10, 15, 30, 50} |
| EMA decay `--ema` | {0, 0.25, 0.5, 0.75, 0.99, 1.0} |
| Codebook size `--num_codebook` | {64, 128, 256, 512, 1024} |
| Balance `--energy_lambda` | {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0} |
| Sharpness `--energy_r` | {1, 2, 4, 6} |
| Warm-up down-weighting `--beta` | {0.05, 0.1, 0.2, 0.5, 1.0} |

The paper's Figure 3 reports the *selected* values for the MF backbone on MIND and Yelp2018 only
(codebook size, EMA decay, balance, sharpness, warm-up length):

| Hyperparameter | MIND (MF) | Yelp2018 (MF) |
|---|--:|--:|
| `--num_codebook` | 1024 | 512 |
| `--ema` | 0.99 | 0.25 |
| `--energy_lambda` | 0.9 | 0.75 |
| `--energy_r` | 4 | 2 |
| `--begin_adv` | 10 | 10 |

## 📎 Citation
If you find this work useful, please cite:
```bibtex
T.B.D
```
