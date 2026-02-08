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
pip install trl
pip install deepspeed
pip install pillow
pip install tensorboard
pip install qwen-vl-utils
pip install peft
pip install transformers==4.57.0
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn --no-build-isolation  # bug
```

## :rocket: Getting Started

### 1. [Supervised Fine-Tuning (SFT)](sft/README.md)

``` bash
cd sft

# sft (data format: examples/data_sft_image_train.json)
bash sft_image.sh

# infer (data format: examples/data_sft_image_test.json)
bash infer_image.sh
```

### 2. Direct Preference Optimization (DPO)

### 3. Group Relative Policy Optimization (GRPO)
