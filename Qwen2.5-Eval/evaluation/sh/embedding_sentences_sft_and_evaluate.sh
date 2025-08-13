# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-cot"
export CUDA_VISIBLE_DEVICES="0"
MAX_TOKENS="100"

# CHECKPOINTS_DIR=... # replace with your own path for storing checkpoints
CHECKPOINTS_DIR="/root/One-Shot-RLVR"


###### DSR-sub #######
PROJECT_NAME="verl_few_shot"
SUB_EXPERIMENT_NAMES=("dsr_sub_sft" "dsr_sub_sft_lr")

GLOBAL_STEP_LIST=($(seq 0 9 45))

# # Loop through each step in the list
for SUB_EXPERIMENT_NAME in "${SUB_EXPERIMENT_NAMES[@]}"; do
    for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
        python embedding_sentences.py \
            --input ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${SUB_EXPERIMENT_NAME}/logic_sentences/global_step_${GLOBAL_STEP}/test_qwen25-math-cot_-1_seed0_t0.7_s0_e-1_logic_sentences.jsonl
        python analyze_prompt_similarities.py ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${SUB_EXPERIMENT_NAME}/logic_sentences/global_step_${GLOBAL_STEP}/embeddings.pt
    done
    
    python analyze_temporal_similarities.py \
        "${CHECKPOINTS_DIR}/${PROJECT_NAME}/${SUB_EXPERIMENT_NAME}/logic_sentences/global_step_*/embeddings.pt" \
        --save-prefix ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${SUB_EXPERIMENT_NAME}/logic_sentences/temporal_analysis \
        --plot-every-n-steps 9
    python animate_pca_evolution.py \
        "${CHECKPOINTS_DIR}/${PROJECT_NAME}/${SUB_EXPERIMENT_NAME}/logic_sentences/global_step_*/embeddings.pt" \
        --output ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${SUB_EXPERIMENT_NAME}/logic_sentences/pca_animation.gif \
        --fps 1 --dpi 150 --figsize 12 10
done

###### DSR-full #######
PROJECT_NAME="verl_few_shot"
FULL_EXPERIMENT_NAMES=("dsr_full_sft" "dsr_full_sft_lr")

GLOBAL_STEP_LIST=($(seq 0 314 1570))

# # Loop through each step in the list
for FULL_EXPERIMENT_NAME in "${FULL_EXPERIMENT_NAMES[@]}"; do
    for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
        python embedding_sentences.py \
            --input ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${FULL_EXPERIMENT_NAME}/logic_sentences/global_step_${GLOBAL_STEP}/test_qwen25-math-cot_-1_seed0_t0.7_s0_e-1_logic_sentences.jsonl
        python analyze_prompt_similarities.py ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${FULL_EXPERIMENT_NAME}/logic_sentences/global_step_${GLOBAL_STEP}/embeddings.pt
    done
    
    python analyze_temporal_similarities.py \
        "${CHECKPOINTS_DIR}/${PROJECT_NAME}/${FULL_EXPERIMENT_NAME}/logic_sentences/global_step_*/embeddings.pt" \
        --save-prefix ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${FULL_EXPERIMENT_NAME}/logic_sentences/temporal_analysis \
        --plot-every-n-steps 314
    python animate_pca_evolution.py \
        "${CHECKPOINTS_DIR}/${PROJECT_NAME}/${FULL_EXPERIMENT_NAME}/logic_sentences/global_step_*/embeddings.pt" \
        --output ${CHECKPOINTS_DIR}/${PROJECT_NAME}/${FULL_EXPERIMENT_NAME}/logic_sentences/pca_animation.gif \
        --fps 1 --dpi 150 --figsize 12 10
done
