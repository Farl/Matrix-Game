<p align="center">
<h1 align="center">Matrix-Game 2.0</h1>
<h3 align="center">An Open-Source, Real-Time, and Streaming Interactive World Model</h3>
</p>

<font size=7><div align='center' >  [[🤗 HuggingFace](https://huggingface.co/Skywork/Matrix-Game-2.0)] [[📖 Technical Report](https://matrix-game-v2.github.io/static/pdf/report.pdf)] [[🚀 Project Website](https://matrix-game-v2.github.io/)] </div></font>

https://github.com/user-attachments/assets/336b0d4a-64f5-4e5c-9b60-6212ddb261c0

## 📝 Overview
**Matrix-Game-2.0** is an interactive world foundation model for real-time long video generation.  Built upon an auto-regressive diffusion-based image-to-world framework, it can generate real-time[25fps] long videos conditioned on keyboard and mouse inputs, enabling fine-grained control and dynamic scene evolution.

## 🤗 Matrix-Game-2.0 Model
we provide three pretrained model weights including universal scenes, GTA driving scene and TempleRun game scene. Please refer to our HuggingFace page to reach these resources.

## Requirements
We tested this repo on the following setups:

### Original Setup (CUDA)
* Nvidia GPU with at least 24 GB memory (A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

### Apple Silicon Support 🍎 (New!)
* **Apple Silicon Mac** (M1/M2/M3/M4)
* **macOS** with Metal Performance Shaders (MPS)
* **16GB+ RAM** (Unified memory)
* **Python 3.8+** with PyTorch 2.0+ (MPS support)
* **60GB+ free disk space** for models

✅ **Fully tested and verified on Apple Silicon!** See [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md) for details.

## Installation

### For CUDA Systems (Original)
Create a conda environment and install dependencies:
```bash
conda create -n matrix-game-2.0 python=3.10 -y
conda activate matrix-game-2.0
# install apex and FlashAttention
# Our project also depends on [FlashAttention](https://github.com/Dao-AILab/flash-attention)
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game-2
pip install -r requirements.txt
python setup.py develop
```

### For Apple Silicon (M1/M2/M3/M4) 🍎
```bash
# Clone this Apple Silicon adapted version
git clone [your-repo-url] Matrix-Game-Working
cd Matrix-Game-Working

# Install dependencies (FlashAttention optional - will auto-fallback)
pip install -r requirements.txt
python setup.py develop

# Download models automatically
bash download_models.sh
```

**Note for Apple Silicon users**: This version includes intelligent fallbacks for Flash Attention and automatic MPS precision management. No CUDA required!


## Quick Start

### Download Models

#### For CUDA Systems
```bash
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
```

#### For Apple Silicon 🍎
```bash
# Use the automated script (recommended)
bash download_models.sh

# Or manual download
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0 --local-dir-use-symlinks
```

For detailed model setup instructions, see [MODELS_SETUP.md](MODELS_SETUP.md).

### Inference

#### For CUDA Systems
```bash
python inference.py \
    --config_path configs/inference_yaml/{your-config}.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --img_path {path-to-the-input-image} \
    --output_folder outputs \
    --num_output_frames 150 \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder}
```

#### For Apple Silicon 🍎
```bash
# Basic inference with optimized settings
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python inference.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --img_path demo_images/universal/0000.png \
    --output_folder outputs \
    --num_output_frames 10 \
    --seed 42 \
    --pretrained_model_path Matrix-Game-2.0
```

### Streaming Inference

#### For CUDA Systems
```bash
python inference_streaming.py \
    --config_path configs/inference_yaml/{your-config}.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --output_folder outputs \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder}
```

#### For Apple Silicon 🍎
```bash
# Interactive streaming with Apple Silicon optimizations
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python inference_streaming.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0
```

### Tips
- In the current version, upward movement for camera may cause brief rendering glitches (e.g., black screens). A fix is planned for future updates. Adjust movement slightly or change direction to resolve it.


## ⭐ Acknowledgements

We would like to express our gratitude to:

- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) for their strong base model
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module
- [MineRL](https://github.com/minerllabs/minerl) for their excellent gym framework
- [Video-Pre-Training](https://github.com/openai/Video-Pre-Training) for their accurate Inverse Dynamics Model

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
```
