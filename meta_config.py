import argparse
import torch
import os

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='RS Models')

parser.add_argument("--project", type=str, default="PRIDE", help="Project name")
parser.add_argument('--seed', type=int, default=2024, help='seed')

parser.add_argument('--model', type=str, default='MF', help='model')

# dataset
parser.add_argument('--dataset', type=str, default='MIND', help='dataset')
parser.add_argument('--min_interaction', type=int, default=10, help='Min interactions')
parser.add_argument('--noise', type=float, default=0.0, help='noise ratio')
parser.add_argument('--add_p', type=float, default=1.0, help='noise ratio')

# model
parser.add_argument('--out_dim', type=int, default=64, help='Output size of Adapter')

# experiment
parser.add_argument('--use_gpu', type=str2bool, default=True, help='training device')
parser.add_argument('--device', type=str, default='gpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--test_batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--val_interval', type=int, default=1, help='Validation interval')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight for L2 loss on basic models.')
parser.add_argument('--min_epochs', type=int, default=1, help='min epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='max epoch')


parser.add_argument('--rec_top_k', type=list, default=[20, 50], help='K in evaluation')


parser.add_argument('--method', type=str, default='PRIDE', help='method of denoising')

parser.add_argument('--temp', type=float, default=0.1, help='noise ratio')
parser.add_argument('--item_num', type=int, default=5, help='max epoch')

# Parameter for PRIDE
parser.add_argument('--num_codebook', type=int, default=512, help='size of codebook')
parser.add_argument('--num_hirearchy', type=int, default=1, help='number of codebook')
parser.add_argument('--begin_adv', type=int, default=15, help='warm_up')
parser.add_argument('--ema', type=float, default=0.75, help='ema')
parser.add_argument(
    '--weight_mode',
    type=str,
    default='lambda_power',
    help='PRIDE weighting mode: [noise_energy_boltzmann, reliability_boltzmann, disagreement_aware, power_product, weighted_geometric_mean, lambda_power]'
)
parser.add_argument('--energy_r', type=float, default=4.0, help='sharpness for lambda_power mode (r > 0)')
parser.add_argument('--energy_lambda', type=float, default=0.5, help='balance for lambda_power mode: 0=stability only, 1=intent only, range (0,1)')
parser.add_argument('--energy_gamma', type=float, default=1.0, help='gamma for energy-based PRIDE weighting (noise_energy_boltzmann / disagreement_aware)')
parser.add_argument('--lambda_dis', type=float, default=1.0, help='disagreement penalty weight for disagreement_aware mode')
parser.add_argument('--tau', type=float, default=1.0, help='temperature for reliability_boltzmann mode')
parser.add_argument('--weight_eps', type=float, default=1e-8, help='epsilon for power_product mode')
parser.add_argument('--wgm_alpha', type=float, default=0.5, help='α for weighted_geometric_mean mode: w = w_intent^α * w_stability^(1-α)')
parser.add_argument('--lambda_mix', type=float, default=0.5,
    help='mixing coefficient for PRIDE full mode: w = s * (λ·c + (1-λ)). '
         '0=s only, 1=s*c (original), range [0.0, 1.0]')

# Parameter for MoE gate
parser.add_argument('--gate_tau', type=float, default=1.0, help='temperature for softmin gate routing (lower = sharper target distribution)')
parser.add_argument(
    '--use_original_weighting_after_warmup',
    type=str2bool,
    default=False,
    help='if true, PRIDE uses the original (non-MoE) weighting after warm-up'
)

# Parameter for R-CE
parser.add_argument('--beta', type=float, default=0.1, help='beta for r-ce weighting')

# Parameter for T-CE
parser.add_argument('--drop_rate', type=float, default=0.2, help='drop rate for t-ce')
parser.add_argument('--num_gradual', type=int, default=30000, help='number of gradual')

# Parameter for BOD
parser.add_argument('--alpha', type=float, default=1.0, help='alpha for BOD')
parser.add_argument('--gamma', type=float, default=1.0, help='gamma for BOD')

# Parameter for DCF
parser.add_argument('--relabel_ratio', type=float, default=0.03, help='ratio for relabeling')
parser.add_argument('--co_lambda', type=float, default=0.01, help='coefficient for consistency loss')
parser.add_argument('--mean_loss_interval', type=int, default=2, help='time step')

# Ablation study
parser.add_argument('--ablation', type=str, default="full", help='ablation study setting: [full, wo_pride, wo_user_intent, wo_stability]')

# Logging
parser.add_argument("--wandb", action="store_true", help="Use W&B logging.")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id


if torch.cuda.is_available() and args.use_gpu:
    print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')