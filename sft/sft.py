import json
import re

import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
)


def format_example(example):
    messages = []
    image_idx = 0
    video_idx = 0
    special_tokens = ["<image>", "<video>"]

    for msg in example["messages"]:
        message = {
            "role": msg["role"],
            "content": []
        }

        content = msg["content"]
        pattern = '|'.join(map(re.escape, special_tokens))
        subcontent_list = re.split(f'({pattern})', content)

        for subcontent in subcontent_list:
            if len(subcontent) == 0:
                continue
            if subcontent == "<image>":
                message["content"].append({"type": "image", "image": example["images"][image_idx]})
                image_idx = image_idx + 1
            elif subcontent == "<video>":
                message["content"].append({"type": "video", "video": example["videos"][video_idx]})
                video_idx = video_idx + 1
            else:
                message["content"].append({"type": "text", "text": subcontent})
        messages.append(message)

    return {"messages": messages}


class DataCollator:
    def __init__(
        self,
        processor,
        mode="train",
        image_size_hw=[256, 256],
        max_length=2048,  # not used now
    ):
        self.processor = processor
        self.mode = mode
        self.image_size_hw = image_size_hw
        self.max_length = max_length

    def __call__(self, batch):
        """
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<video>\n Would you mind generating the radiology report for the specified chest CT scan?"
                },
                {
                    "role": "assistant",
                    "content": "Findings: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Impression:  A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung."
                }
            ],
            "videos": [
                "/hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_npy/valid/vaild_1/vaild_1_a/valid_1_a_1.npy"
            ]
        }
        """
        # prepare for processor
        batch_format = []
        for example in batch:
            batch_format.append(format_example(example))

        """
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": "/hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_npy/valid/vaild_1/vaild_1_a/valid_1_a_1.npy"
                        },
                        {
                            "type": "text",
                            "text": "Would you mind generating the radiology report for the specified chest CT scan?"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Findings: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Impression:  A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung."
                        }
                    ]
                }
            ]
        }
        """
        # prepare for model
        texts = []
        images = []
        videos = []

        for example in batch_format:
            texts.append(self.processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False if self.mode == "train" else True,
            ))
            image_inputs, video_inputs = process_vision_info(example["messages"])
            if image_inputs is not None:
                # images.extend(image_inputs)  # list: PIL.Image.Image, [W, H]
                for image in image_inputs:
                    image = image.resize((self.image_size_hw[1], self.image_size_hw[0]))
                    images.append(image)
            if video_inputs is not None:
                videos.extend(video_inputs)  # list: [T, 3, H, W], [0, 255]

        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        batch_processed = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )  # input_ids, attention_mask, pixel_values, image_grid_thw
        if self.mode != "train":
            return batch_processed

        labels = torch.full_like(batch_processed["input_ids"], -100)

        # supervise answer + eos_token
        B, L = batch_processed["input_ids"].shape
        for input_ids_cur, labels_cur in zip(batch_processed["input_ids"], labels):
            start_idx = 0
            end_idx = 0
            while start_idx < L:
                if input_ids_cur[start_idx] == self.processor.tokenizer.encode("<|im_start|>")[0]:
                    if input_ids_cur[start_idx + 1] == self.processor.tokenizer.encode("assistant")[0]:
                        start_idx = start_idx + len(self.processor.tokenizer.encode("<|im_start|>assistant\n"))
                        end_idx = start_idx + 1
                        while input_ids_cur[end_idx] != self.processor.tokenizer.encode("<|im_end|>")[0]:
                            end_idx = end_idx + 1
                        labels_cur[start_idx:end_idx+1] = input_ids_cur[start_idx:end_idx+1]
                start_idx = start_idx + 1

        # mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # mask vision tokens
        # <|vision_start|> <|vision_end|> <|image_pad|> <|video_pad|>
        vision_token_ids = [151652, 151653, 151655, 151656]
        for vision_token_id in vision_token_ids:
            labels[labels == vision_token_id] = -100

        # shift happens in transformers/loss/loss_utils/ForCausalLMLoss
        batch_processed["labels"] = labels
        return batch_processed


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, TrainingArguments))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Model
    model_kwargs = dict(
        dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Data
    # dataset = load_dataset("json", data_files=script_args.dataset_name)
    with open(script_args.dataset_name, "r") as f:
        data_list = json.load(f)
    dataset = Dataset.from_list(data_list)
    if training_args.eval_strategy != "no":
        splits = dataset.train_test_split(test_size=0.1)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    data_collator = DataCollator(processor)

    # Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"] if training_args.eval_strategy != "no" else dataset,
        eval_dataset=splits["test"] if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
        processing_class=processor,  # for saving processor config
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
