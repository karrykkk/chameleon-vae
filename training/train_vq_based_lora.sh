export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint='localhost:5666' \
    --master_port 10003 \
    train_vq_based_lora.py \
    --base_model_path /liymai24/sjtu/siqi/leloykun/ckpt/anole-7b-lelo \
    --dataset textvqa \
    --output_dir /liymai24/sjtu/siqi/leloykun/outputs/tvqa_vq_based_lora64_bs8 \
    --report_to wandb \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --save_steps 1000 \
    --fp16 False \
    --save_strategy "steps" \
    --logging_steps 1 \
    --bf16 True \
    --lora_r 64 \
    --lora_alpha 128 \
    --deepspeed /liymai24/sjtu/siqi/leloykun/training/ds_config.json \
    --gradient_accumulation_steps 4 \
    # --gradient_checkpointing True \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --do_train \
    # --lr_scheduler_type "cosine" \
    # --eval_strategy "no" \
    # --save_total_limit 5 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --lr_scheduler_type "cosine" \
    # --logging_steps 30 \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'ChameleonDecoderLayer' \
    # # --ddp_find_unused_parameters True \
    # --deepspeed /liymai24/sjtu/siqi/leloykun/training/ds_config.json \