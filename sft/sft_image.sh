# ~18GB
# export CUDA_VISIBLE_DEVICES=0
# python sft.py \
#     --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
#     --torch_dtype bfloat16 \
#     --attn_implementation flash_attention_2 \
#     --dataset_name dev/sft_image_train.json \
#     --remove_unused_columns False \
#     --output_dir work_dirs/sft-image-single-gpu \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --bf16 True \
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

# ~13GB * 2
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 --master_port 29501 sft.py \
    --deepspeed zero2_offload.json \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name dev/sft_image_train.json \
    --remove_unused_columns False \
    --output_dir work_dirs/sft-image-multi-gpus \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 True \
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
