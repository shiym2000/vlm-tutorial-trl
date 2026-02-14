# ZeRO2: ~20GB*4
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config_file deepspeed_zero2.yaml \
    sft.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --dataset_name dev/sft_image_train.json \
    --image_size_h 512 \
    --image_size_w 512 \
    --tune_encoder full \
    --tune_connector full \
    --tune_llm full \
    --output_dir work_dirs/sft-image \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 True \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing True \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 5 \
    --per_device_eval_batch_size 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False
    # --resume_from_checkpoint work_dirs/sft-image/checkpoint-50

# ZeRO2: ~10GB*4
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch \
#     --config_file deepspeed_zero2.yaml \
#     sft.py \
#     --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
#     --dtype bfloat16 \
#     --attn_implementation sdpa \
#     --use_peft True \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules q_proj k_proj v_proj o_proj \
#     --lora_task_type CAUSAL_LM \
#     --dataset_name dev/sft_image_train.json \
#     --image_size_h 512 \
#     --image_size_w 512 \
#     --tune_encoder full \
#     --tune_connector full \
#     --tune_llm lora \
#     --remove_unused_columns False \
#     --output_dir work_dirs/sft-image-lora \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --bf16 True \
#     --max_length 2048 \
#     --dataloader_num_workers 8 \
#     --gradient_checkpointing True \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --report_to tensorboard \
#     --save_strategy steps \
#     --save_steps 5 \
#     --save_total_limit 2 \
#     --eval_strategy steps \
#     --eval_steps 5 \
#     --per_device_eval_batch_size 2 \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --greater_is_better False
#     # --resume_from_checkpoint work_dirs/sft-image-lora/checkpoint-50
