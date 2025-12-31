import argparse
import json
import os
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TextStreamer,
)

from sft import DataCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    # Model
    model_kwargs = dict(
        dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    # Data
    with open(args.dataset_name, "r") as f:
        data_list = json.load(f)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    data_collator = DataCollator(processor, mode="infer")
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Inference
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
                    streamer=streamer,
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
                # print(f"{output_text}\n")  # not used because of streamer

        output_list.append(example_cur)

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
