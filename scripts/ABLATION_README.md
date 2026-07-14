# MF Ablation Study — REQUIEM Weight Components

## Overview

Ablation study for the REQUIEM recommendation system (MF backbone).
3 variants × 4 datasets × 1 seed = **12 total runs**.

| Variant | `ablation` flag | Description |
|---|---|---|
| Full model | `full` | All components active (results from HP tuning) |
| w/o User Intent Weight | `wo_user_intent` | User intent weight fixed to 1 |
| w/o Stability Weight | `wo_stability` | Stability weight fixed to 1 |
| w/o both weights | `wo_requiem` | Both weights fixed to 1 (`torch.ones_like`) |

All variants use the same optimal hyperparameters found for the full model.

---

## Prerequisites

**Full model Stage 1–4 hyperparameter tuning must be complete** before running ablations.

You need the best values for each dataset from:

| Stage | Parameters |
|---|---|
| Stage 1 | `begin_adv`, `ema`, `num_codebook` |
| Stage 2 | `energy_r`, `energy_lambda` |
| Stage 3 | `beta` |
| Stage 4 | `drop_rate`, `num_gradual` |

---

## Injecting Best Params into YAML Files

Each YAML in `config/ablation/` contains `# PLACEHOLDER` comments on the parameters that need updating. Search for them:

```bash
grep -n "PLACEHOLDER" config/ablation/wo_user_intent_yelp.yaml
```

Edit each file and replace the placeholder values with the actual Full model best params per dataset. The `# NOTE: unused` lines in `wo_requiem_*.yaml` do not need to be changed — those params are ignored at runtime.

Example (Yelp `wo_user_intent`):
```yaml
begin_adv:
  values: [<stage1_best>]   # was 30 (placeholder)
ema:
  values: [<stage1_best>]   # was 1.0 (placeholder)
num_codebook:
  values: [<stage1_best>]   # was 256 (placeholder)
energy_r:
  values: [<stage2_best>]   # was 2 (placeholder)
energy_lambda:
  values: [<stage2_best>]   # was 0.5 (placeholder)
beta:
  values: [<stage3_best>]   # was 0.1 (placeholder)
drop_rate:
  values: [<stage4_best>]   # was 0.1 (placeholder)
num_gradual:
  values: [<stage4_best>]   # was 10000 (placeholder)
```

Repeat for all 12 YAML files (3 variants × 4 datasets).

---

## Running Experiments

### Individual variant

```bash
# from project root
bash scripts/run_ablation_wo_user_intent.slurm
bash scripts/run_ablation_wo_stability.slurm
bash scripts/run_ablation_wo_requiem.slurm
```

Each script runs 4 datasets sequentially (~4 hours per variant).

### All variants sequentially (~12 hours)

```bash
bash scripts/run_all_ablations.slurm
```

### All variants in parallel (~4 hours total)

```bash
bash scripts/run_ablation_wo_user_intent.slurm &
bash scripts/run_ablation_wo_stability.slurm &
bash scripts/run_ablation_wo_requiem.slurm &
wait
echo "All done."
```

---

## Checking Results in WandB

All ablation runs log to project **`MF-ABLATION`**.

1. Open WandB → Projects → `MF-ABLATION`
2. Filter runs by `ablation` tag to compare variants side-by-side
3. Key metrics: `valid_Recall@20`, `valid_Recall@50`, `valid_NDCG@20`, `valid_NDCG@50`

To compare with the Full model, add the `MF-HParam` or Full model project runs to the same WandB report.

---

## Interpreting Results

For each dataset, compare against the Full model baseline:

| Metric drop | Interpretation |
|---|---|
| `Full` > `wo_user_intent` | User intent weight contributes positively |
| `Full` > `wo_stability` | Stability weight contributes positively |
| `Full` > `wo_requiem` | Both weights together contribute positively |
| `wo_user_intent` ≈ `wo_requiem` | Stability weight drives most of the gain |
| `wo_stability` ≈ `wo_requiem` | User intent weight drives most of the gain |

---

## File Structure

```
config/ablation/
  wo_user_intent_yelp.yaml
  wo_user_intent_mind.yaml
  wo_user_intent_amazon_book.yaml
  wo_user_intent_toys.yaml
  wo_stability_yelp.yaml
  wo_stability_mind.yaml
  wo_stability_amazon_book.yaml
  wo_stability_toys.yaml
  wo_requiem_yelp.yaml
  wo_requiem_mind.yaml
  wo_requiem_amazon_book.yaml
  wo_requiem_toys.yaml

scripts/
  run_ablation_wo_user_intent.slurm
  run_ablation_wo_stability.slurm
  run_ablation_wo_requiem.slurm
  run_all_ablations.slurm
  ABLATION_README.md  ← this file
```

---

## Extending to LightGCN

When ready to extend ablations to the LightGCN backbone:

1. Duplicate each YAML and set `model: ['LightGCN']`
2. Update dataset-specific params (lr, weight_decay, begin_adv, etc.) with LGN Full model best
3. Create `run_ablation_lgn_*.slurm` scripts following the same pattern
4. Consider using a separate WandB project (e.g. `LGN-ABLATION`) for clean separation
