MODEL_PATH=work_dirs/sft-image-multi-gpus/checkpoint-116

# ~4GB
export CUDA_VISIBLE_DEVICES=0
python infer.py \
    --model_name_or_path $MODEL_PATH \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name dev/sft_image_test.json \
    --output_path $MODEL_PATH/infer/output.json \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0
