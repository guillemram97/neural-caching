from utils import (
    neptune_log,
)
import pdb


def update_online_metrics(avg_stats, stats):
    for key, value in stats.items():
        if key in avg_stats.keys() and key != "performance":
            avg_stats[key] += value
        else:
            avg_stats[key] = value
    avg_stats["n_points"] += 1
    return avg_stats


def log_test(run, score_gold, score_llm, step):
    stats = {"real_score_gold": score_gold, "real_score_llm": score_llm}
    neptune_log(
        run=run,
        pref=f"test/",
        stats=stats,
        epoch=step,
    )


def log_final(run):
    neptune_log(
        run=run,
        pref=f"final/",
        stats={"finished": 1},
        epoch=0,
    )
    return


def reset_avg_online_metrics(stats):
    avg_stats = {}
    for key in stats.keys():
        avg_stats[key] = 0
    avg_stats["n_points"] = 0
    return avg_stats


def get_online_metrics_mult(args, metric, sample, pred, decision, budgets, performance):
    stats = {"performance": performance}
    for idx, b in enumerate(budgets):
        stats[str(b) + "-dec"] = decision[idx]

    suffix = "hard"
    if args.soft_labels:
        suffix = "soft"

    target_ref = [list(sample["llm_" + suffix]), list(sample["gold_" + suffix])]
    for idx, b in enumerate(budgets):
        pred_b = pred[idx]
        for name, ref in zip(["llm", "gold"], target_ref):
            metric.reset()
            metric.add_batch(
                predictions=list(pred_b),
                references=list(ref),
            )
            online_metrics = metric.compute()
            for idx, online_metric in enumerate(online_metrics):
                if idx == 0:
                    stats[f"{b}-{name}_{idx}"] = online_metric
    return stats


# def get_online_metrics_single(args, metric, sample, pred, decision, BT_pred, EN_pred):
#    stats = {"decision": decision, "EN_pred": EN_pred}
#    if args.soft_labels:
#        suffix = "soft"
#        stats["BT_pred"] = BT_pred
#    else:
#        suffix = "hard"
#    target_ref = [list(sample["llm_" + suffix]), list(sample["gold_" + suffix])]
#    for name, ref in zip(["llm", "gold"], target_ref):
#        metric.reset()
#        metric.add_batch(
#            predictions=list(pred),
#            references=list(ref),
#        )
#        online_metrics = metric.compute()
#        for idx, online_metric in enumerate(online_metrics):
#            if idx == 0:
#                stats[f"{name}_{idx}"] = online_metric
#    return stats


def log_avg_online(run, avg_stats, step, b):
    stats = {}
    for key, value in avg_stats.items():
        if key != "n_points" and key.split("-")[0] == str(b):
            stats["avg_" + key] = avg_stats[key] / avg_stats["n_points"]

    neptune_log(
        run=run,
        pref=f"online/",
        stats=stats,
        epoch=step,
    )
    return
