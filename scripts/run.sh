#!/bin/sh
source scripts/cluster.sh
# Neptune tags 
TAGS="${TAGS:=debug}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:=no}"

TASK_NAME="${TASK_NAME:=isear}"
BASE_MODEL="${BASE_MODEL:=t5-base}"
BUDGET="${BUDGET:=1000}"
COST_EXT="${COST_EXT:=1}"
RETRAIN_FREQ="${RETRAIN_FREQ:=100}"
SOFT_LABELS="${SOFT_LABELS=1}"
TARGET="${TARGET:=llm}"
STRATEGY="${STRATEGY:=b1}"
N_INIT="${N_INIT:=100}"
CHECKPOINT="${CHECKPOINT:=-1}"
P_STRAT="${P_STRAT:=0}"
ORACLE="${ORACLE:=0}"
ORACLE_BT="${ORACLE_BT:=0}"
TEMPERATURE="${TEMPERATURE:=1.0}"
MAX_LEN="${MAX_LEN:=512}"
MAX_OUT_LEN="${MAX_OUT_LEN:=2}" # WE LIMIT FOR CLASSIFICATION
NUM_BEAMS="${NUM_BEAMS:=1}"
ONLY_IMPROVE="${ONLY_IMPROVE:=0}"
ACTIVE="${ACTIVE:=no}"
LR="${LR:=0.0005}"
BATCH="${BATCH:=16}"
BATCH_EVAL="${BATCH_EVAL:=16}"
EPOCHS="${EPOCHS:=30}" # Keep it fixed
WEIGHT_DECAY="${WEIGHT_DECAY:=1e-2}"
IGNORE_LLM="${IGNORE_LLM:=0}"

# Don't limit the number of examples and check how long it takes to execute a
# full run, maybe it takes less than a day and we're fine, if it's not then
# change the code and save checkpoint
TRAIN_SAMPLES="${TRAIN_SAMPLES:=10000}"
EVAL_SAMPLES="${EVAL_SAMPLES=100}"
TEST_SAMPLES="${TEST_SAMPLES=10000}"

# Early stopping
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:=2}"
EARLY_STOP="${EARLY_STOP:=5}"

R="${R:=16}" 
LORA_SCALING="${LORA_SCALING:=0.25}"


SEED="${SEED=0}"

python -m main \
  --model_name_or_path $BASE_MODEL \
  --task_name $TASK_NAME \
  --budget $BUDGET \
  --n_init $N_INIT \
  --p_strat $P_STRAT \
  --cost_ext $COST_EXT \
  --checkpoint $CHECKPOINT \
  --save_checkpoint $SAVE_CHECKPOINT \
  --retrain_freq $RETRAIN_FREQ \
  --soft_labels $SOFT_LABELS \
  --temperature $TEMPERATURE \
  --target $TARGET \
  --active $ACTIVE \
  --oracle $ORACLE \
  --oracle_BT $ORACLE_BT \
  --strategy $STRATEGY \
  --max_length $MAX_LEN \
  --only_improve $ONLY_IMPROVE \
  --per_device_train_batch_size $BATCH \
  --per_device_eval_batch_size $BATCH_EVAL \
  --learning_rate $LR \
  --ignore_llm $IGNORE_LLM \
  --num_train_epochs $EPOCHS \
  --train_samples $TRAIN_SAMPLES \
  --eval_samples $EVAL_SAMPLES \
  --test_samples $TEST_SAMPLES \
  --eval_every_epochs $EVAL_EVERY_EPOCHS \
  --early_stop $EARLY_STOP \
  --r $R \
  --lora_scaling $LORA_SCALING \
  --num_beams $NUM_BEAMS \
  --weight_decay $WEIGHT_DECAY \
  --tags $TAGS \
  --max_out_length $MAX_OUT_LEN \
  --seed $SEED