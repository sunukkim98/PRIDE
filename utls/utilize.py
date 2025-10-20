import os
import random
import sys

import numpy as np
import torch

import torch.nn.functional as F

from dotmap import DotMap
from petname import english
import hashlib

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_run(log_path, args, seed=None):
    global original_stdout, original_stderr, outfile

    if seed is not None:
        set_seed(seed)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    f = open(os.path.join(log_path, f"log_{args.noise}_{seed}.txt"), 'w')
    f = Unbuffered(f)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    outfile = os.path.join(log_path, f"log_{args.noise}_{seed}.txt")

    # sys.stderr = f
    # sys.stdout = f

def restore_stdout_stderr():
    global original_stdout, original_stderr, outfile

    sys.stdout = original_stdout
    sys.stderr = original_stderr

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
    


def slice_lists(list1, list2, batch_size):
    len1, len2 = len(list1), len(list2)
    
    num_batches = -(-max(len1, len2) // batch_size)  # 使用负数进行整除来实现向上取整
    
    slice_size1 = -(-len1 // num_batches)
    slice_size2 = -(-len2 // num_batches)
    
    slices1 = [list1[i:i + slice_size1] for i in range(0, len1, slice_size1)]
    slices2 = [list2[i:i + slice_size2] for i in range(0, len2, slice_size2)]
    
    while len(slices1) < num_batches:
        slices1.append([])
    while len(slices2) < num_batches:
        slices2.append([])

    return slices1, slices2


def batch_split(users, batch_size):
    for i in range(0, len(users), batch_size):
        yield users[i:i + batch_size]

class RunNameGenerator:
    """Singleton class to generate unique names based on hyperparameter hashes."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensures only one instance of RunNameGenerator exists."""
        if not cls._instance:
            cls._instance = super(RunNameGenerator, cls).__new__(cls)
        return cls._instance

    def generate_name(self, hyperparams: DotMap) -> str:
        """Generate a unique name based on a hash derived from hyperparameters."""

        def _generate_petname() -> str:
            """Generate a random adjective-name combination."""
            adjective = random.choice(english.adjectives)
            name = random.choice(english.names)
            return f"{adjective}-{name}"
        
        # Convert hyperparameters to a sorted string representation
        hparam_str = "_".join(
            [f"{k}={v}" for k, v in sorted(hyperparams.items(), key=lambda x: x[0])]
        )

        # Generate a hash integer from the hyperparameter string
        hash_int = int(hashlib.sha256(hparam_str.encode()).hexdigest(), 16)

        # Preserve the current random state to avoid affecting global randomness
        current_state = random.getstate()

        # Seed the random generator with the hash value
        random.seed(hash_int)
        words = _generate_petname()
        number = random.randint(0, 999)

        # Restore the original random state
        random.setstate(current_state)

        return f"{words}-{number}"
