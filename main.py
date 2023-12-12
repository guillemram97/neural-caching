from utils import (
    parse_args,
    setup_basics,
    neptune_log,
    set_seeds,
)
from utils.online_logs import (
    update_online_metrics,
    reset_avg_online_metrics,
    get_online_metrics_mult,
    log_avg_online,
    log_test,
    log_final,
)
import numpy as np
from metrics import Metric
from handler import handler_LLM
from student import student
from accelerate import Accelerator
from accelerate.logging import get_logger
from task import (
    get_task,
    make_datacollator,
)
import pdb
import copy
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()
    accelerator = Accelerator()
    run = setup_basics(accelerator, logger, args)

    # Pre-Logging
    run["args"] = vars(args)
    set_seeds(args.seed)

    task = get_task(
        accelerator=accelerator,
        args=args,
        model=None,
    )
    if not task.is_classification:
        args.is_classification = False
    else:
        args.soft_labels = (
            True  # for classification, we always use a soft labels objective
        )
    online_dataloader = task.data["online_dataloader"]
    st = student(args, task, run, accelerator)
    budgets = [int(b) for b in args.budget.split(",")]

    wrap = handler_LLM(args, st, task)
    metric = Metric(args, soft=args.soft_labels, online=True)

    # Initialize student model
    # If we put a checkpoint, we load the model and we skip the first $checkpoint steps
    if args.checkpoint != "-1":
        PATH = "checkpoints/" + args.task_name + "/" + str(args.checkpoint) + ".pt"
        if args.n_init == 100 and args.strategy == "MV":
            PATH = (
                "checkpoints/"
                + args.task_name
                + "/"
                + str(args.checkpoint.split("_")[0])
                + "_500.pt"
            )
        st.init_checkpoint(PATH)
        wrap = handler_LLM(args, st, task)
        wrap.student_vec = []
        if args.strategy == "MV":
            for idx in range(5):
                st_aux = student(args, task, run, accelerator)
                aux_name = int(args.checkpoint.split("_")[1])
                if args.n_init == 100:
                    aux_name = 500
                PATH_AUX = (
                    "checkpoints/"
                    + args.task_name
                    + "/"
                    + str(args.checkpoint.split("_")[0])
                    + "_"
                    + str(aux_name - 400 + 100 * idx)
                    + ".pt"
                )
                st_aux.init_checkpoint(PATH_AUX)
                wrap.student_vec.append(copy.deepcopy(st_aux.model).cpu())
                del st_aux

    stop_retraining = args.strategy == "EM_raw"
    send_update = False

    for step, sample in enumerate(online_dataloader):

        if args.checkpoint != "-1" and step < args.n_init:
            wrap.save_cache(sample)
            if args.strategy == "CS":
                wrap.output = wrap.call_llm(sample)
                wrap.obtain_embed(sample)
                wrap.save_embed()

        if args.checkpoint == "-1" or step >= args.n_init:
            gc.collect()
            decision, pred = wrap.query(sample)

            stats = get_online_metrics_mult(
                args,
                metric,
                sample,
                pred,
                decision,
                budgets,
                wrap.performance,
            )
            neptune_log(
                run=run,
                pref=f"online/",
                stats=stats,
                epoch=step,
            )
            if step == 0 or (args.checkpoint != "-1" and step == args.n_init):
                avg_online = reset_avg_online_metrics(stats)
            avg_online = update_online_metrics(avg_online, stats)

            if wrap.retrain or (
                step + 1 and (step + 1) % args.retrain_freq == 0 and not stop_retraining
            ):
                set_seeds(args.seed)
                wrap.BT = []
                cache = wrap.retrieve_cache()
                train_dataloader, eval_dataloader = make_datacollator(
                    args, task.tokenizer, cache
                )
                train_dataloader, eval_dataloader = accelerator.prepare(
                    train_dataloader, eval_dataloader
                )

                if wrap.retrain:
                    st.suffixes.append(str(budgets[len(wrap.budget_models)]) + "-")
                st.train(train_dataloader, eval_dataloader)

                del train_dataloader, eval_dataloader
                if step + 1 and (step + 1) % args.retrain_freq == 0: wrap.update = False

                wrap.reorder_students()
                if wrap.budget_arr[-1] == 0:
                    stop_retraining = True
                    wrap.delete_cache()
                send_update = True

            if send_update or step == len(online_dataloader) - 1:
                log_avg_online(run, avg_online, step, budgets[-1])
                avg_online = reset_avg_online_metrics(stats)
                send_update = False
                if step == len(online_dataloader) - 1:
                    log_final(run)

    if run is not None:
        run.stop()


if __name__ == "__main__":
    main()
