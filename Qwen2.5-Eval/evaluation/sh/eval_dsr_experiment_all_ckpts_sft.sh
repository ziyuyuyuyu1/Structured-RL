# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-cot"
export CUDA_VISIBLE_DEVICES="0"
MAX_TOKENS="3072"

# CHECKPOINTS_DIR=... # replace with your own path for storing checkpoints
CHECKPOINTS_DIR="/root/One-Shot-RLVR"


###### DSR-sub #######
PROJECT_NAME="verl_few_shot"
EXPERIMENT_NAMES=("dsr_sub_sft" "dsr_sub_sft_lr")
GLOBAL_STEP_LIST=($(seq 0 9 45))

# # Loop through each step in the list
for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"; do
    for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
        echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
        MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}
        OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
        bash sh/generate_math_answer.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
    done
done

EXPERIMENT_NAMES=("dsr_full_sft" "dsr_full_sft_lr")
GLOBAL_STEP_LIST=($(seq 0 314 1570))

# # Loop through each step in the list
for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"; do
    for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
        echo "======== Evaluating checkpoint at global step: ${GLOBAL_STEP} ========"
        MODEL_NAME_OR_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}
        OUTPUT_DIR=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/eval/global_step_${GLOBAL_STEP}
        bash sh/generate_math_answer.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MAX_TOKENS $OUTPUT_DIR
    done
done




