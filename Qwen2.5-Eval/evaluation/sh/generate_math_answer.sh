
set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKENS_PER_CALL=$3
OUTPUT_DIR=$4

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAMES="math500"
IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

for DATASET in "${DATASETS[@]}"; do
    if [ ! -d "${OUTPUT_DIR}/${DATASET}" ] || [ -z "$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "============ All datasets in ${DATA_NAMES} already evaluated ============="
    for DATASET in "${DATASETS[@]}"; do
        METRICS_FILE=$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)
        echo "============ ${DATASET} ============="
        cat ${METRICS_FILE}
    done
else
    TOKENIZERS_PARALLELISM=false \
    python3 -u off_the_track_evaluation.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAMES} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0.7 \
        --n_sampling 500 \
        --top_p 0.95 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
        --pipeline_parallel_size 1 \
        --overwrite 
fi

