import argparse

from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--merged_model_path", type=str, default=None)
    args = parser.parse_args()

    # Save processor
    print(f"[1/4] Saving processor: {args.lora_weights_path} -> {args.merged_model_path}...")
    processor = AutoProcessor.from_pretrained(args.lora_weights_path)
    processor.save_pretrained(args.merged_model_path)

    # Load model
    print(f"[2/4] Loading model: {args.model_name_or_path}...")
    model_kwargs = dict(
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    # Merge LoRA weights
    print(f"[3/4] Merging LoRA weights: {args.lora_weights_path}...")
    model = PeftModel.from_pretrained(model, args.lora_weights_path)
    model = model.merge_and_unload()

    # Save merged model
    print(f"[4/4] Saving merged model: {args.merged_model_path}...")
    model.save_pretrained(args.merged_model_path)
