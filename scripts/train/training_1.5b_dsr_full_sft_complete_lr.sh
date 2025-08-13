#!/bin/bash
set -x

CHECKPOINTS_DIR=checkpoints/dsr_full_sft_lr

export VLLM_ATTENTION_BACKEND=XFORMERS

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/train/one_shot_rlvr/dsr_full_sft_simple.parquet \
    data.val_files=data/test/dsr_val_simple.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=4096 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    optim.betas=[0.9,0.95] \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$CHECKPOINTS_DIR \
    trainer.project_name='verl_few_shot' \
    trainer.experiment_name='Qwen2.5-Math-1.5B-dsr_sub-sft-complete-lr' \
    trainer.total_epochs=5 \
    trainer.logger=['console','swanlab'] \
    use_remove_padding=True \
    trainer.default_hdfs_dir=null 2>&1 | tee verl_sft_demo_complete_lr.log 