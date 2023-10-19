import torch
import pdb
import os
import json
from argparse import Namespace
import evaluate
from task import get_task
from metrics import Metric
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, Softmax

from utils import set_seeds

from transformers import T5ForConditionalGeneration


cross_entropy = CrossEntropyLoss()
softmax = Softmax()


def get_model(args):
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
    )
    return model


def load_optimizer(model, args):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "alphas" not in n],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    return optimizer


def finish_training(accelerator, model, eval_metric, args):
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )


# de moment nomes generar prob de un token
def soft_loss(logits, soft_labels, temperature=1):
    # logits es BS x LENGTH x VOCAB
    # soft labels seran probs de 3-4 tokens en classification tasks
    # en altres tasks seran altres tokens ?
    cross_batch = 0
    for idx, label in enumerate(soft_labels):
        logits[idx] = softmax(logits[idx])
        label = softmax(label / temperature)
        cross_batch += cross_entropy(logits[idx], label)
    return cross_batch / logits.shape[0]


# de moment nomes generar prob de un token
def soft_loss_weighted(logits, soft_labels, temperature=1):
    # afegir weights
    cross_batch = 0
    for idx, label in enumerate(soft_labels):
        logits[idx] = softmax(logits[idx])
        label = softmax(label / temperature)
        factor = 1
        if label[1] > label[0]:
            factor = 10
        cross_batch += factor * cross_entropy(logits[idx], label)
    return cross_batch / logits.shape[0]


def train_epoch(
    model,
    train_dataloader,
    accelerator,
    lr_scheduler,
    optimizer,
    args,
    dic_classes=None,
):
    model.train()
    set_seeds(args.seed)
    total_loss = 0

    losses = []

    freq = 100

    for step, batch in enumerate(train_dataloader):
        if args.target == "gold":
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.gold_hard,
            )
        else:
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.llm_hard,
            )

        if args.soft_labels:
            if args.target == "gold":
                loss = soft_loss(
                    outputs[1][:, 0, dic_classes].cpu(),
                    batch.gold_soft.cpu().float(),
                    args.temperature,
                )
            else:
                loss = soft_loss(
                    outputs[1][:, 0, dic_classes].cpu(),
                    batch.llm_soft.cpu(),
                    args.temperature,
                )
        else:
            loss = outputs.loss

        total_loss += loss.detach().float().item()
        losses.append(loss.detach().float().item())

        accelerator.backward(loss)

        if step % freq == 0:
            print(f"loss = {sum(losses) / len(losses)}")
            losses = []

        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

    return total_loss


def evaluate_model(
    model, accelerator, eval_dataloader, metric, args, dic_classes, target
):
    model.eval()
    samples_seen = 0

    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            if metric.soft:
                predictions = model.generate(
                    **{
                        "input_ids": batch["input_ids"].cuda(),
                        "attention_mask": batch["attention_mask"].cuda(),
                    },
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                predictions = list(np.array(predictions[1][0].cpu())[:, dic_classes])

                # fix a formatting bug
                if type(batch[target + "_soft"]) == torch.Tensor:
                    batch[target + "_soft"] = [aux for aux in batch[target + "_soft"]]
                predictions, references = accelerator.gather(
                    (predictions, batch[target + "_soft"])
                )
            else:
                predictions = model.generate(
                    **{
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                    },
                    num_beams=args.num_beams,
                    max_length=args.max_out_length,
                    decoder_start_token_id=model.model.config.bos_token_id,
                )
                predictions, references = accelerator.gather(
                    (predictions, batch[target + "_hard"])
                )

        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()

    if isinstance(eval_metric, dict):
        _, eval_metric = list(eval_metric.items())[0]
        # eval_metric = sum(eval_metric.values()) / len(eval_metric)

    return eval_metric


def save_model(accelerator, epoch, args):
    output_dir = f"epoch_{epoch}"
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, output_dir)
    accelerator.save_state(output_dir)


def get_hparams(args, task_name):
    PATH_PARAMS = "hparams/" + task_name + "/params.json"
    if os.path.exists(PATH_PARAMS):
        f = open(PATH_PARAMS)
        data = json.load(f)
        aux_args = vars(args)
        for hparam in data:
            aux_args[hparam] = data[hparam]
        args = Namespace(**aux_args)

    return args
