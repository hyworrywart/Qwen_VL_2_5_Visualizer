# 🔍 Qwen2.5-VL Attention Visualization Tool

A comprehensive tool for visualizing attention patterns in Qwen2.5-VL vision-language model during text generation. This tool allows you to see exactly which parts of an image the model focuses on when generating each token.

![Project Banner](https://img.shields.io/badge/Model-Qwen2.5--VL-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM for 3B model)
- 16GB+ RAM

## 🚀 Installation

### 1. Clone the Repository

```bash
cd Qwen_VL_2_5_Visualizer
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n qwen_viz python=3.10
conda activate qwen_viz

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 📖 Usage

### Quick Start

```bash
python app.py
```

Then open your browser and navigate to: `http://127.0.0.1:7861`


Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Attention extraction
EXTRACT_ALL_LAYERS = True  # Or specify SPECIFIC_LAYERS

# Visualization
HEATMAP_COLORMAP = "jet"  # 'hot', 'viridis', 'plasma', etc.
HEATMAP_ALPHA = 0.5  # Transparency (0-1)

# Memory optimization
STORE_ON_CPU = True  # Offload attention to CPU
USE_FLOAT16 = True  # Use half precision
```

## 🔧 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Qwen2.5-VL Model                        │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  Vision Encoder  │────────▶│  Language Model   │        │
│  │  (Patch Embed +  │         │  (Decoder Layers) │        │
│  │   Transformer)   │         │                   │        │
│  └──────────────────┘         └──────────────────┘        │
│           │                            │                    │
│           │                     ┌──────▼──────┐           │
│           │                     │  Attention  │◀──────────┤
│           │                     │   Weights   │  Hooks    │
│           │                     └─────────────┘           │
└───────────┼──────────────────────────┬───────────────────┘
            │                          │
            ▼                          ▼
    ┌───────────────┐         ┌──────────────────┐
    │ Image Patches │         │ Attention Maps   │
    │  Coordinates  │         │  (per token)     │
    └───────┬───────┘         └────────┬─────────┘
            │                          │
            └────────────┬─────────────┘
                         ▼
              ┌──────────────────────┐
              │  Visualization       │
              │  (Heatmap Overlay)   │
              └──────────────────────┘
```

### Key Components

1. **Attention Extractor** (`attention_extractor.py`)
   - Registers forward hooks on decoder attention modules
   - Captures attention weights during generation
   - Stores per-layer, per-head attention

2. **Attention Processor** (`attention_processor.py`)
   - Maps vision tokens to image patch positions
   - Aggregates attention across layers/heads
   - Creates 2D attention maps

3. **Visualizer** (`visualization.py`)
   - Generates heatmap overlays
   - Supports multiple colormaps and transparency
   - Creates comparison views

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team**: For the amazing Qwen2.5-VL model
- **Hugging Face**: For Transformers library
- **Gradio**: For the easy-to-use interface framework

## 📚 References

- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

