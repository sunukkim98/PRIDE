import os
from dotmap import DotMap
import wandb
import wandb.errors
from utls import utilize

class Monitor:
    """Handles WandB initialization, configuration, and logging as a Singleton."""
    _instance = None

    def __new__(cls, args=None):
        if cls._instance is None:
            cls._instance = super(Monitor, cls).__new__(cls)
            cls._instance._initialize(args)
        return cls._instance
    
    def _initialize(self, args):
        """Initialize WandB and retrieve hyperparameters."""
        self.args = args
        self._init_wandb()
        self.set_hyperparams()

    def _init_wandb(self):
        """Initialize WandB (login & project setup) without setting hyperparameters."""
        if self.args.wandb is False:
            print("🚀 Running without WandB. WandB is disabled.")
            return
        
        try:
            wandb.login()

            log_dir = "./log/wandb"
            os.makedirs(log_dir, exist_ok=True)

            if not wandb.run:
                wandb.init(project=self.args.project, dir=log_dir)
            print("✅ WandB initialized successfully.")
        except wandb.errors.CommError:
            print("⚠️ WandB login failed. Running in disabled mode.")
            wandb.init(mode="disabled")

    def set_hyperparams(self):
        """Set hyperparameters from command-line args or Wandb (if sweep)."""
        if self.args.wandb is False:
            config = self.args
        else:
            if wandb.run and bool(wandb.config.as_dict()) is False:
                print("📌 WandB detected but not a sweep. Using input arguments.")
                wandb.config.update(vars(self.args))
            config = wandb.config

        self.hyperparams = DotMap({
            "seed": config.seed,
            "model": config.model,
            "dataset": config.dataset,
            "out_dim": config.out_dim,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "max_epochs": config.n_epochs,
            "method": config.method,
            "num_codebook": config.num_codebook,
            "warm_up": config.begin_adv,
            "ema": config.ema
        })

        if wandb.run:
            rng = utilize.RunNameGenerator()
            wandb.run.name = rng.generate_name(self.hyperparams)
            wandb.run.save("*.py")
            wandb.run.save("config/*.yaml")
    
    def get_hyperparams(self):
        """Get the stored hyperparameters."""
        return self.hyperparams
    
    def log(self, log_data: dict):
        """Log metrics to WandB."""
        if wandb.run:
            wandb.log(log_data)

    def finish(self):
        """Ensure WandB session is properly closed."""
        if wandb.run:
            wandb.finish()