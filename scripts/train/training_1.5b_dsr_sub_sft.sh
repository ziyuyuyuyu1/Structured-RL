#!/bin/bash
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path

export VLLM_ATTENTION_BACKEND=XFORMERS

# set nproc_per_node * micro_batch_size_per_gpu = 128
NPROC_PER_NODE=VALUE_TO_BE_SET
MICRO_BATCH_SIZE_PER_GPU=VALUE_TO_BE_SET # (128 / nproc_per_node)

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/train/one_shot_rlvr/dsr_sub.parquet \
    data.val_files=data/test/math500.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.max_length=4096 \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=True \
    model.fsdp_config.param_offload=False \
    model.fsdp_config.optimizer_offload=False \
    optim.lr=5e-5 \
    optim.betas=[0.9,0.95] \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$CHECKPOINTS_DIR \
    trainer.project_name='verl_few_shot' \
    trainer.experiment_name='Qwen2.5-Math-1.5B-dsr_sub-sft' \
    trainer.total_epochs=5 \
    trainer.logger=['console','swanlab'] \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.default_hdfs_dir=null 2>&1 | tee verl_sft_demo.log