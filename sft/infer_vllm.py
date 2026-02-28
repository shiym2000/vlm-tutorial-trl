import argparse
import json
import os
from tqdm import tqdm

from transformers import AutoProcessor
from vllm import (
    LLM,
    SamplingParams,
)

from sft import DataCollatorForSFTQwen3VL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--video_size_t", type=int, default=None)
    parser.add_argument("--image_size_h", type=int, default=None)
    parser.add_argument("--image_size_w", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    # Load model
    print(f"[1/6] Loading model: {args.model_name_or_path}...")
    engine_args = dict(
        model=args.model_name_or_path,
        max_model_len=args.max_length,
        max_num_seqs=1,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 128 * 512 * 512,
            "fps": 2,
        },
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    model = LLM(**engine_args)

    # Load dataset
    print(f"[2/6] Loading dataset: {args.dataset_name}...")
    with open(args.dataset_name, "r") as f:
        data_list = json.load(f)

    # Prepare data collator
    print(f"[3/6] Preparing data collator...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSFTQwen3VL(
        processor=processor,
        mode="vllm",
        video_size_t=args.video_size_t,
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
        max_length=args.max_length,
    )

    # Preprocess data
    print(f"[4/6] Preprocessing data...")
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    inputs_vllm = []
    for example in tqdm(data_list):
        messages = example["messages"]
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        example_cur = {"messages": messages}
        if "images" in example:
            example_cur["images"] = example["images"]
        if "videos" in example:
            example_cur["videos"] = example["videos"]
        inputs = data_collator([example_cur])

        if inputs["images"] is not None:
            inputs_vllm.append(
                {
                    "prompt": inputs["texts"][0],
                    "multi_modal_data": {"image": inputs["images"][0]},
                }
            )
        elif inputs["videos"] is not None:
            video = (
                inputs["videos"][0].permute(0, 2, 3, 1).cpu().numpy(),  # TCHW -> THWC
                # {'total_num_frames': 16, 'fps': 25.0, 'duration': 9.72, 'video_backend': 'opencv', 'frames_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'do_sample_frames': False}
                {"total_num_frames": args.video_size_t, "fps": 2.0, "frames_indices": list(range(args.video_size_t)), "do_sample_frames": False},
            )
            inputs_vllm.append(
                {
                    "prompt": inputs["texts"][0],
                    "multi_modal_data": {"video": [video]},
                }
            )

    # Start inference
    print(f"[5/6] Starting inference...")
    outputs = model.generate(
        inputs_vllm,
        sampling_params=sampling_params,
    )

    # Save results
    print(f"[6/6] Saving results: {args.output_path}...")
    output_list = []
    for example, o in tqdm(zip(data_list, outputs), total=len(data_list)):
        messages = example["messages"]
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        messages.append({"role": "assistant", "content": o.outputs[0].text})

        example["messages_inferred"] = messages
        output_list.append(example)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)
