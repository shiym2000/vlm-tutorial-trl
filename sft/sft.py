import json
import re
from dataclasses import dataclass, field
from PIL import Image

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.functional as F
from datasets import Dataset
from torchvision.io import read_video
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)


@dataclass
class ScriptArgumentsForSFT(ScriptArguments):
    video_size_t: int = field(
        default=16,
        metadata={"help": "The number of frames to resize videos to."},
    )
    image_size_h: int = field(
        default=512,
        metadata={"help": "The height to resize input images/videos to."},
    )
    image_size_w: int = field(
        default=512,
        metadata={"help": "The width to resize input images/videos to."},
    )
    tune_encoder: str = field(
        default="freeze",
        metadata={"help": "The tuning strategy for the vision encoder. It can be `freeze` or `full`."},
    )
    tune_connector: str = field(
        default="freeze",
        metadata={"help": "The tuning strategy for the connector. It can be `freeze` or `full`."},
    )
    tune_llm: str = field(
        default="freeze",
        metadata={"help": "The tuning strategy for the LLM. It can be `freeze`, `full` or `lora`."},
    )


class DataCollatorForSFTQwen3VL:
    def __init__(
        self,
        processor,
        mode="train",
        video_size_t=16,
        image_size_h=512,
        image_size_w=512,
        max_length=4096,  # not used now
    ):
        self.processor = processor
        self.mode = mode
        self.video_size_t = video_size_t
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
        self.max_length = max_length

    @staticmethod
    def format_example_swift2trl(example):
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

                if subcontent == "<image>":  # 2D image
                    message["content"].append({"type": "image", "image": example["images"][image_idx]})
                    image_idx = image_idx + 1

                elif subcontent == "<video>":
                    if ".nii.gz" in example["videos"][video_idx]:  # 3D medical image
                        message["content"].append({"type": "video", "image3d": example["videos"][video_idx]})
                    else:  # video
                        message["content"].append({"type": "video", "video": example["videos"][video_idx]})
                    video_idx = video_idx + 1

                else:  # text
                    message["content"].append({"type": "text", "text": subcontent})

            messages.append(message)

        return {"messages": messages}

    @staticmethod
    def temporal_resize(
        video: torch.Tensor,
        target_T: int,
        method: str,
    ) -> torch.Tensor:
        """
        将 (T, C, H, W) 的张量在时间维调整到 target_T。

        参数
        ----
        video     : Tensor  (T, C, H, W)
        target_T  : int     目标帧数
        method    : str
            - 'sample'      均匀抽帧，只支持下采样
            - 'linear'      仅对时间维做 1D 线性插值
        """
        T, C, H, W = video.shape
        if T <= target_T:
            return video  # 不做任何处理

        # ---------- sample ----------
        if method == "sample":
            # 均匀抽帧
            idx = torch.linspace(0, T - 1, target_T, device=video.device).round().long()
            video = video.index_select(0, idx)

        # ---------- 插值 ----------
        elif method == "linear":
            video = video.permute(1, 2, 3, 0).contiguous()      # (C, H, W, T)
            video = video.view(-1, 1, T)                            # (C*H*W, 1, T)
            video = torch.nn.functional.interpolate(video, size=target_T, mode="linear", align_corners=False)  # (C*H*W, 1, target_T)
            video = video.view(C, H, W, target_T).permute(3, 0, 1, 2).contiguous()  # (target_T, C, H, W)

        return video

    def __call__(self, batch):
        """
        {
            "messages": [
                {"role": "user", "content": "<image>\nxxx?"},
                {"role": "assistant", "content": "xxx."},
            ],
            "images": ["path/to/image.jpg"],
        }
        """
        # prepare for processor
        batch_format = []
        for example in batch:
            batch_format.append(self.format_example_swift2trl(example))

        """
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "path/to/image.jpg"},
                        {"type": "text", "text": "\nxxx?"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "xxx."}
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

            for message in example["messages"]:
                for content in message["content"]:
                    if content["type"] == "image":
                        image = Image.open(content["image"]).convert("RGB")
                        image = image.resize((self.image_size_w, self.image_size_h))
                        images.append(image)  # PIL.Image

                    elif content["type"] == "video" and "video" in content:
                        video, audio, info = read_video(content["video"])  # torch.tensor, [T, H, W, 3], 0-255
                        video = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
                        video = self.temporal_resize(
                            video=video.float(),
                            target_T=self.video_size_t,
                            method="sample",
                        )
                        video = F.resize(
                            img=video,
                            size=[self.image_size_h, self.image_size_w],
                        )
                        videos.append(video)  # torch.tensor, [T, 3, H, W], 0-255

                    elif content["type"] == "video" and "image3d" in content:
                        nii = nib.load(content["image3d"])
                        image3d = nii.get_fdata()  # (X, Y, Z) == (W, H, T)

                        # normalize to 0-255
                        HU_MIN, HU_MAX = -1000, 1000
                        image3d = np.clip(image3d, HU_MIN, HU_MAX)
                        image3d = (image3d - HU_MIN) / (HU_MAX - HU_MIN) * 255

                        # resize
                        image3d = torch.from_numpy(image3d).permute(2, 1, 0)  # (T, H, W)
                        image3d = image3d.unsqueeze(1).float()  # (T, 1, H, W)
                        image3d = image3d.repeat(1, 3, 1, 1)  # (T, 3, H, W)
                        image3d = self.temporal_resize(
                            video=image3d,
                            target_T=self.video_size_t,
                            method="linear",
                        )
                        image3d = F.resize(
                            img=image3d,
                            size=[self.image_size_h, self.image_size_w],
                        )
                        videos.append(image3d)  # torch.tensor, [T, 3, H, W], 0-255

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


if __name__ == "__main__":
    parser = TrlParser((ModelConfig, ScriptArgumentsForSFT, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Load model
    print(f"[1/4] Loading model: {model_args.model_name_or_path}...")
    model_kwargs = dict(
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Load dataset
    print(f"[2/4] Loading dataset: {script_args.dataset_name}...")
    with open(script_args.dataset_name, "r") as f:
        data_list = json.load(f)
    dataset = Dataset.from_list(data_list)
    if training_args.eval_strategy != "no":
        splits = dataset.train_test_split(test_size=0.1)

    # Prepare data collator
    print(f"[3/4] Preparing data collator...")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    data_collator = DataCollatorForSFTQwen3VL(
        processor=processor,
        mode="train",
        video_size_t=script_args.video_size_t,
        image_size_h=script_args.image_size_h,
        image_size_w=script_args.image_size_w,
        max_length=training_args.max_length,
    )

    # Start training
    print("[4/4] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"] if training_args.eval_strategy != "no" else dataset,
        eval_dataset=splits["test"] if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
    )
    if not model_args.use_peft:
        for param in trainer.model.parameters():
            param.requires_grad = False
    if script_args.tune_encoder == "full":
        for param in trainer.model.model.visual.parameters():
            param.requires_grad = True
        for param in trainer.model.model.visual.merger.parameters():
            param.requires_grad = False
        if hasattr(trainer.model.model.visual, "deepstack_merger_list"):
            for param in trainer.model.model.visual.deepstack_merger_list.parameters():
                param.requires_grad = False
    if script_args.tune_connector == "full":
        for param in trainer.model.model.visual.merger.parameters():
            param.requires_grad = True
        if hasattr(trainer.model.model.visual, "deepstack_merger_list"):
            for param in trainer.model.model.visual.deepstack_merger_list.parameters():
                param.requires_grad = True
    if script_args.tune_llm == "full":
        for param in trainer.model.model.language_model.parameters():
            param.requires_grad = True
        for param in trainer.model.lm_head.parameters():
            param.requires_grad = True
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
