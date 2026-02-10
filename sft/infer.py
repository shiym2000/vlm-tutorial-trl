import argparse
import json
import os
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    # TextStreamer,
)

from sft import DataCollatorForSFTQwen3VL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--image_size_h", type=int, default=None)
    parser.add_argument("--image_size_w", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    # Load model
    print(f"[1/5] Loading model: {args.model_name_or_path}...")
    model_kwargs = dict(
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    # Load dataset
    print(f"[2/5] Loading dataset: {args.dataset_name}...")
    with open(args.dataset_name, "r") as f:
        data_list = json.load(f)

    # Prepare data collator
    print(f"[3/5] Preparing data collator...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSFTQwen3VL(
        processor=processor,
        mode="infer",
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
        max_length=args.max_length,
    )

    # Start inference
    print(f"[4/5] Starting inference...")
    # streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    output_list = []
    for example in tqdm(data_list):
        # init for multi-turn
        messages = []
        example_cur = {"messages": messages}
        if "images" in example:
            example_cur["images"] = example["images"]
        if "videos" in example:
            example_cur["videos"] = example["videos"]

        for msg in example["messages"]:
            if msg["role"] != "assistant":
                messages.append(msg)
                print(f"[{msg['role']}]:\n{msg['content']}\n")
            else:
                example_cur["messages"] = messages
                inputs = data_collator([example_cur])
                inputs = inputs.to(model.device)

                print("[assistant]:")
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True if args.temperature > 0 else False,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    # streamer=streamer,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                messages.append({"role": "assistant", "content": output_text})
                print(f"{output_text}\n")

        example["messages_inferred"] = messages
        output_list.append(example)

    # Save results
    print(f"[5/5] Saving results: {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)
