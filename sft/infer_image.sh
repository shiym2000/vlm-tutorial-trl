# ~4GB
# export CUDA_VISIBLE_DEVICES=0
# python infer.py \
#     --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
#     --dtype bfloat16 \
#     --attn_implementation flash_attention_2 \
#     --dataset_name dev/sft_image_test.json \
#     --output_path dev/sft_image_test_inferred.json \
#     --image_size_h 512 \
#     --image_size_w 512 \
#     --max_length 2048 \
#     --max_new_tokens 512 \
#     --num_beams 1 \
#     --temperature 0

MODEL_PATH=work_dirs/sft-image/checkpoint-50

export CUDA_VISIBLE_DEVICES=0
python infer.py \
    --model_name_or_path $MODEL_PATH \
    --dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name dev/sft_image_test.json \
    --output_path $MODEL_PATH/infer/output.json \
    --image_size_h 512 \
    --image_size_w 512 \
    --max_length 2048 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0
