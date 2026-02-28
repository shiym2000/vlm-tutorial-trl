# VLM-Tutorial-TRL

A step-by-step tutorial for training vision–language models (VLMs) with [TRL](https://github.com/huggingface/trl).

## :package: Installation

``` bash
# 1. clone and navigate
git clone https://github.com/shiym2000/vlm-tutorial-trl.git
cd vlm-tutorial-trl

# 2. create a conda environment, activate it and install packages
conda create -n trl python=3.10
conda activate trl
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.57.0
pip install trl==0.24.0
pip install deepspeed
pip install pillow
pip install tensorboard
pip install qwen-vl-utils
pip install peft
pip install vllm==0.11.0
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3
pip install nibabel
```

To train a 3D medical VLM based on Qwen3-VL, following modifications are required:

``` python
# envs/trl/lib/python3.10/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py (L208)
metadata.fps = 2 if metadata.fps is None else metadata.fps

# envs/trl/lib/python3.10/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py (L161)
metadata.fps = 2
```

## :rocket: Getting Started

### 1. [Supervised Fine-Tuning (SFT)](sft/README.md)

``` bash
cd sft

# 1. image
bash sft_image.sh  # sft (data format: examples/data_sft_image_train.json)
bash infer_image.sh  # infer (data format: examples/data_sft_image_test.json)

# 2. 3D medical image
bash sft_image3d.sh  # sft (data format: examples/data_sft_image3d_train.json)
bash infer_image3d.sh  # infer (data format: examples/data_sft_image3d_test.json)
```

### 2. Direct Preference Optimization (DPO)

### 3. Group Relative Policy Optimization (GRPO)
