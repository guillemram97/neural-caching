import numpy as np
import string
import re
import torch
from collections import Counter
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr
from rouge import Rouge
from torch.nn import CrossEntropyLoss, Softmax
import pdb
from transformers import T5Tokenizer

cross_entropy = CrossEntropyLoss()
softmax = Softmax()

METRICS = {
    "rt-polarity": ["ACC"],
    "isear": ["ACC"],
    "openbook": ["ACC"],
    "fever": ["ACC"],
}


class Metric:
    """
    It's used only for dev / test evaluation
    Not during training
    """

    def __init__(self, args, soft=False, online=False):
        self.task_name = args.task_name
        self.tokenizer = T5Tokenizer.from_pretrained(
            args.model_name_or_path, model_max_length=args.max_length
        )
        self.predictions = []
        self.references = []
        self.online = online
        self.args = args
        self.soft = soft

    def reset(self):
        self.predictions = []
        self.references = []

    def add_batch(self, predictions, references):
        if not self.soft:
            if type(predictions[0]) != str:
                predictions = self.tokenizer.batch_decode(
                    predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            if type(references[0]) != str:
                references = self.tokenizer.batch_decode(
                    references,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
        self.predictions = self.predictions + predictions
        self.references = self.references + references

    def compute(self):
        if self.soft:
            metrics = evaluate_soft(
                self.predictions, self.references, self.args.temperature
            )
        else:
            metrics = evaluate_hard(self.predictions, self.references, self.task_name)
        self.reset()
        return metrics


def evaluate_soft(predictions, data, temperature=1):
    cross_entropy_score = []
    accuracy = []

    for idx, pred in enumerate(predictions):
        predictions[idx] = softmax(torch.tensor(pred).cuda().float())
        data[idx] = softmax(torch.tensor(data[idx]).cuda().float() / temperature)
        cross_entropy_score.append(cross_entropy(predictions[idx], data[idx]))
        predictions[idx].argmax() == data[idx].argmax()
        accuracy.append(1 * (predictions[idx].argmax() == data[idx].argmax()).tolist())
    return [sum(accuracy) / len(data), sum(cross_entropy_score) / len(data)]


def evaluate_hard(predictions, data, task):
    metrics = METRICS[task]
    tmp_metrics = []
    assert len(predictions) == len(data)

    if "ACC" in metrics:
        accs = []
        for prediction, dp in zip(predictions, data):
            accs.append(get_accruacy_over_list(prediction, dp))
        tmp_metrics.append(np.mean(accs))
    if "hatexplain" in metrics:
        accs = []
        for prediction, dp in zip(predictions, data):
            accs.append(get_accuracy_hatexplain(prediction, dp))
        tmp_metrics.append(np.mean(accs))
    if "QA-F1" in metrics:
        f1s = []
        for prediction, dp in zip(predictions, data):
            f1s.append(get_f1_over_list(prediction, dp))
        tmp_metrics.append(np.mean(f1s))
    if "Classification-F1" in metrics:
        if isinstance(data[0], list):
            data = [dat[0] for dat in data]
        return tmp_metrics.append(f1_score(data, predictions, average="macro"))
    return tmp_metrics


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def accuracy(prediction, ground_truth):
    return prediction.lower() == ground_truth.lower()


def get_accuracy_hatexplain(prediction, groundtruth):
    prediction = prediction.split(" ")
    groundtruth = groundtruth.split(" ")
    precision = 0
    f1 = None
    if len(prediction[1:]):
        for pred in prediction[1:]:
            precision += get_accruacy_over_list(pred, groundtruth[1:])
        precision = precision / len(prediction[1:])
        f1 = precision
    recall = 0
    if len(groundtruth[1:]):
        for truth in groundtruth[1:]:
            recall += get_accruacy_over_list(truth, prediction[1:])
        recall = recall / len(groundtruth[1:])
        if f1 is None:
            f1 = recall
        else:
            f1 = 0.5 * precision + 0.5 * recall
    if f1 is None:
        return accuracy(prediction[0], groundtruth[0])
    return 0.5 * accuracy(prediction[0], groundtruth[0]) + 0.5 * f1


def get_accruacy_over_list(prediction, groundtruth):
    if isinstance(groundtruth, list):
        if len(groundtruth) == 0:
            return 0
        return np.max([accuracy(prediction, gt) for gt in groundtruth])
    return accuracy(prediction, groundtruth)


def get_f1_over_list(prediction, groundtruth):
    if isinstance(groundtruth, list):
        if len(groundtruth) == 0:
            return 0
        return np.max([qa_f1_score(prediction, gt) for gt in groundtruth])
    return qa_f1_score(prediction, groundtruth)


def get_exact_match_over_list(prediction, groundtruth):
    if isinstance(groundtruth, list):
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match_over_list(prediction, gt) for gt in groundtruth])
    return normalize_answer(prediction) == normalize_answer(groundtruth)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_trivial_white_space(text):
        while len(text) and text[0] == " ":
            text = text[1:]
        while len(text) and text[-1] == " ":
            text = text[:-1]
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(
        remove_trivial_white_space(remove_articles(remove_punc(lower(s))))
    )
