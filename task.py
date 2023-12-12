import os
from transformers import DataCollatorForSeq2Seq, T5Tokenizer
from datasets import arrow_dataset
from torch.utils.data import DataLoader
import datasets
from torch.utils.data import Subset
import torch
from functools import partial
import pandas as pd
import ast
import json
import pdb


class Task:
    def __init__(self, task_name, tokenizer, soft_labels):
        self.task_name = task_name
        DATA_DIR = os.getenv("DATA_PATH")
        self.path = os.path.join(DATA_DIR, task_name)
        self.tokenizer = tokenizer
        self.load_config()
        self.load_data()
        self.soft_labels = soft_labels

    def load_config(self):
        with open(os.path.join(self.path, "config.json")) as f:
            config = json.load(f)
        self.is_classification = config["is_classification"]
        if self.is_classification:
            self.classes = config["classes"]
            self.soft_classes = config["soft_classes"]
        self.data_path = config["training_data"]

    def load_data(self):
        data = {}
        for split, split_path in self.data_path.items():
            fin = pd.read_csv(
                os.path.join(self.path, split_path),
            )
            if self.task_name == "ag_news":
                fin = pd.read_csv(os.path.join(self.path, split_path), nrows=10000)
                if split == "test":
                    fin = pd.read_csv(os.path.join(self.path, split_path), nrows=1000)
            inputs = list(fin["input"].values.astype(str))
            gold_hard = list(fin["gold_hard"].values.astype(str))
            if "llm_soft" in fin.columns:
                llm_soft = list(fin["llm_soft"].values.astype(str))
            if "llm_hard" in fin.columns:
                llm_hard = list(fin["llm_hard"].values.astype(str))

            data[split] = arrow_dataset.Dataset.from_dict(
                {
                    "inputs": inputs,
                    "gold_hard": gold_hard,
                    "llm_hard": llm_hard,
                    "llm_soft": llm_soft,
                }
            )
        self.raw_data = datasets.DatasetDict(data)

    def load_classes(self):
        self.classes_dict = {}
        self.classes_dict_gold = {}
        for idx, class_name in enumerate(self.classes):
            target = self.tokenizer.encode(class_name, add_special_tokens=False)[0]
            self.classes_dict[self.soft_classes[idx]] = target
            self.classes_dict_gold[class_name] = target
        return

    def preprocess(self, accelerator, args, model=None):
        def process_data_to_model_inputs(is_eval: bool, batch):
            out = {}
            # Tokenizer will automatically set [BOS] <text> [EOS]
            out["input_ids"] = self.tokenizer(
                batch["inputs"],
                padding=False,
                max_length=args.max_length,
                truncation=True,
            ).input_ids

            if self.is_classification:
                out["gold_soft"] = make_soft(batch["gold_hard"], target="gold")
                if not self.soft_labels:
                    out["llm_soft"] = make_soft(batch["llm_hard"], target="llm")
                else:
                    out["llm_soft"] = select_classes(batch["llm_soft"])
                    
            if is_eval:
                out["gold_hard"] = batch["gold_hard"]
                out["llm_hard"] = batch["llm_hard"]
            else:
                # limited to max_out_length
                out["gold_hard"] = self.tokenizer(
                    batch["gold_hard"],
                    padding=False,
                    max_length=args.max_out_length,
                    truncation=True,
                ).input_ids
                out["llm_hard"] = self.tokenizer(
                    batch["llm_hard"],
                    padding=False,
                    max_length=args.max_out_length,
                    truncation=True,
                ).input_ids
            return out

        def collate_for_eval(default_collate, batch):
            inputs = [{"input_ids": x["input_ids"]} for x in batch]
            out = default_collate(inputs)
            out["llm_hard"] = [x["llm_hard"] for x in batch]
            out["gold_hard"] = [x["gold_hard"] for x in batch]
            if self.is_classification:
                out["llm_soft"] = [x["llm_soft"] for x in batch]
                out["gold_soft"] = [x["gold_soft"] for x in batch]
            return out

        def select_classes(batch_soft_labels):
            new_batch = []
            for soft_labels in batch_soft_labels:
                soft_labels = ast.literal_eval(soft_labels)
                soft_labels = soft_labels[0]
                new_soft_labels = []
                for key in self.soft_classes:
                    if key in soft_labels:
                        new_soft_labels.append(soft_labels[key])
                    else:
                        new_soft_labels.append(-100)
                new_batch.append(new_soft_labels)
            return new_batch

        def make_soft(batch_hard_labels, target):
            if target == "gold":
                classes_dict = self.classes_dict_gold
            else:
                classes_dict = self.classes_dict
            new_batch = []
            for hard_label in batch_hard_labels:
                new_soft_labels = []
                for label in classes_dict.keys():
                    if label == hard_label:
                        new_soft_labels.append(0)
                    else:
                        new_soft_labels.append(-100)
                new_batch.append(new_soft_labels)
            return new_batch

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=model, padding="longest"
        )
        eval_collator = partial(collate_for_eval, data_collator)

        processed_data = {}

        for split in self.data_path.keys():
            max_samples = getattr(args, f"{split}_samples")
            self.raw_data[split] = random_subset(
                dataset=self.raw_data[split],
                max_samples=max_samples,
                seed=args.seed,
            )

            self.raw_data[split] = arrow_dataset.Dataset.from_list(
                list(self.raw_data[split])
            )
            processed_data[split] = self.raw_data[split].map(
                partial(process_data_to_model_inputs, split in ["test"]),
                batched=True,
                batch_size=args.per_device_eval_batch_size,
                remove_columns=self.raw_data[split].column_names,
            )

        online_dataloader = DataLoader(
            processed_data["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=1,
        )

        test_dataloader = DataLoader(
            processed_data["test"],
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        idx_wrong = []
        idx_right = []
        for idx in range(len(processed_data["test"])):
            tgt = (processed_data["test"]["gold_soft"][idx]).index(
                max(processed_data["test"]["gold_soft"][idx])
            )
            llm_pred = (processed_data["test"]["llm_soft"][idx]).index(
                max(processed_data["test"]["llm_soft"][idx])
            )
            if tgt != llm_pred:
                idx_wrong.append(idx)
            else:
                idx_right.append(idx)

        test_wrong = processed_data["test"].select(idx_wrong)
        test_wrong_dataloader = DataLoader(
            test_wrong,
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        online_dataloader, test_dataloader, test_wrong_dataloader = accelerator.prepare(
            online_dataloader,
            test_dataloader,
            test_wrong_dataloader,
        )

        self.data = {
            "online_dataloader": online_dataloader,
            "test_dataloader": test_dataloader,
            "test_wrong_dataloader": test_wrong_dataloader,
        }
        return


def random_subset(dataset, max_samples: int, seed: int = 42):
    if max_samples >= len(dataset) or max_samples == -1:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, perm[:max_samples].tolist())


def get_task(accelerator, args, model=None):
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length
    )

    # load config, data, and preprocess
    task = Task(args.task_name, tokenizer, args.soft_labels)
    if task.is_classification:
        task.load_classes()
    task.preprocess(accelerator, args, model=None)
    return task


def make_datacollator(args, tokenizer, processed_data, model=None):
    processed_data = arrow_dataset.Dataset.from_dict(processed_data)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    aux = processed_data.train_test_split(test_size=0.1)
    train_dataloader = DataLoader(
        aux["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        aux["test"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, eval_dataloader
