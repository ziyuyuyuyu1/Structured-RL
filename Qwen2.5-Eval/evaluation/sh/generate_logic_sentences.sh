
set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKENS_PER_CALL=$3
OUTPUT_DIR=$4

SPLIT="test"
NUM_TEST_SAMPLE=-1

ALL_EXIST=true

if [ ! -d "${OUTPUT_DIR}" ] || [ -z "$(find ${OUTPUT_DIR} -name '*.jsonl' -print -quit)" ]; then
    ALL_EXIST=false
fi


if [ "$ALL_EXIST" = true ]; then
    echo "============ All logic sentences already generated ============="
    METRICS_FILE=$(find ${OUTPUT_DIR} -name '*.jsonl' -print -quit)
    echo "============ ${METRICS_FILE} ============="
    cat ${METRICS_FILE}
else
    echo "============ Generating logic sentences ============="
    TOKENIZERS_PARALLELISM=false \
    python3 -u logic_structure_sample_generation.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0.7 \
        --n_sampling 1000 \
        --top_p 0.95 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
        --pipeline_parallel_size 1 \
        --overwrite 
fi

