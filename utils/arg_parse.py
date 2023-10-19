import argparse
from transformers import SchedulerType
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="The name of the task to train on.",
    )
    parser.add_argument(
        "--soft_labels",
        type=int,
        default=1,
        help="Whether soft labels are provided.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="llm",
        help="Are we using labels from the llm or gold for training the student.",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default="no",
        help="Default is no.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="b1",
        help="What API call strategy is followed.",
    )
    parser.add_argument(
        "--p_strat", type=float, help="Hyperparameter for the strategy."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for soft labels (softmax)",
    )
    parser.add_argument(
        "--only_improve",
        type=int,
        default=0,
        help="If 1, we don't keep a new model if it lowers accuracy",
    )
    parser.add_argument(
        "--active",
        type=str,
        default="no",
    )
    parser.add_argument(
        "--oracle",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--oracle_BT",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ignore_llm",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=100,
        help="Number of initial API calls we do in any strategy.",
    )
    parser.add_argument(
        "--max_length",
        required=True,
        type=int,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--max_out_length",
        type=int,
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        required=True,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        required=True,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="-1",
        help="Initialise student from a given checkpoint number.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
    )
    parser.add_argument(
        "--early_stop",
        type=int,
    )
    parser.add_argument(
        "--r",
        type=int,
    )
    parser.add_argument(
        "--perm",
        type=int,
    )
    parser.add_argument(
        "--lora_scaling",
        type=float,
    )
    parser.add_argument(
        "--budget",
        type=str,
        help="Can be a list if strategy is NOT MV.",
    )
    parser.add_argument(
        "--cost_ext",
        type=int,
    )
    parser.add_argument(
        "--retrain_freq",
        type=int,
    )
    parser.add_argument(
        "--is_classification",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Neptune tags. String delimited by ,",
    )

    args = parser.parse_args()

    return args
