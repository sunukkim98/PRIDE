from meta_config import args
from utls.model_config import *
from utls.trainer import *
from utls.utilize import init_run, restore_stdout_stderr
from monitor import Monitor

def main(seed=2024, main_file=""):
    args.seed = seed
    path = f"./log/{args.dataset}/{args.model}/{args.method}/{main_file}/"
    
    init_run(log_path=path, args=args, seed=args.seed)

    glo = globals()
    global_config = dict(args)
    global_config["main_file"] = main_file
    print(global_config)
   
    global_config["model_config"] = glo[f"get_{global_config['model']}_config"](global_config)
    global_config["model_config"]["denoise_config"] = glo[f"get_{global_config['method']}_config"](global_config)
    print(global_config["model"])
    print(global_config["model_config"])
    global_config['checkpoints'] = 'checkpoints'

    if global_config["model"] == "NeuMF":
        embedding_size = global_config["model_config"]["dim"]
        mlp_dim = global_config["model_config"]["layer_sizes"][0] // 2
        global_config["out_dim"] = embedding_size + mlp_dim
    else:
        global_config["out_dim"] = global_config["model_config"]["dim"]

    trainer_name = "CFTrainer" if global_config["method"] == "Origin" else f"{global_config['method']}CFTrainer"
    trainer =  glo[trainer_name](global_config)
    trainer.train()
    
    restore_stdout_stderr()


if __name__ == '__main__':
    monitor = Monitor(args)
    args = monitor.get_hyperparams()
    times = 1
    main_file = datetime.now().strftime('%Y%m%d%H')
    main_file = f"{main_file}_{args.lr}_{args.weight_decay}_{args.begin_adv}_{args.ema}_{args.num_codebook}_3"
    for t in range(times):
        main(seed=2024+t, main_file=main_file)
