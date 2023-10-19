import logging
import os
import datasets
import transformers
import copy
import torch
import random
import numpy as np

from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from accelerate.utils import set_seed
import neptune.new as neptune
import pdb


def setup_basics(accelerator, logger, args):
    compatibility_check()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    tags = [str(item) for item in args.tags.split(",")]
    if tags == ["''"]:
        tags = [""]
    PROJECT_NAME = os.getenv("PROJECT_NAME_NEPTUNE")
    API_TOKEN_NEPTUNE = os.getenv("API_TOKEN_NEPTUNE")
    run = neptune.init_run(
        project=PROJECT_NAME,
        tags=tags,  # custom_run_id,
        api_token=API_TOKEN_NEPTUNE,
    )

    return run


def neptune_log(run, pref, stats, epoch):
    for k, v in stats.items():
        run[f"{pref}{k}"].log(v, step=epoch)


def compatibility_check():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.24.0.dev0")
    require_version(
        "datasets>=1.8.0",
        "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
    )


def set_seeds(seed: int = 0, is_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic:  # this can slow down performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    return


class EarlyStopper:
    def __init__(self, stop_epochs, save_mem=False):
        self.best_val = -1
        self.counter = 0
        self.best_model = None
        self.save_mem = save_mem
        self.stop_epochs = stop_epochs

    def update(self, eval_metric, model):
        """
        eval_metric: bigger -> better
        """
        self.counter += 1
        if eval_metric > self.best_val:
            self.counter = 0
            self.best_val = eval_metric
            self.best_model = copy.deepcopy(model)

    def should_finish(self):
        """
        Will stop after stop_epochs with no improvement
        """
        return self.counter >= self.stop_epochs

    def get_best(self):
        return self.best_model
