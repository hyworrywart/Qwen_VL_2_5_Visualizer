"""
Configuration file for Qwen2.5-VL Attention Visualization Tool
"""

import os
import torch
from pathlib import Path

# =============================================================================
# Model Configuration
# =============================================================================

# Model identifiers
MODEL_NAME = "/data/yuwei_hu/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/Qwen2.5-VL-7B-Instruct"
PROCESSOR_NAME = "/data/yuwei_hu/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/Qwen2.5-VL-7B-Instruct"  # Processor is compatible across sizes

# Model loading parameters
MODEL_DEVICE_MAP = "auto"  # Auto-assign layers to available devices
MODEL_TORCH_DTYPE = torch.float32  # Use float32 for full precision

# Special token IDs (from Qwen2.5-VL config)
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

# Vision configuration (from Qwen2.5-VL architecture)
VISION_PATCH_SIZE = 14  # Each patch is 14x14 pixels
SPATIAL_MERGE_SIZE = 2  # 2x2 patches merged into one token
TEMPORAL_PATCH_SIZE = 2  # For videos
VISION_HIDDEN_SIZE = 3584  # Vision encoder hidden dimension
TEXT_HIDDEN_SIZE = 3584  # Text model hidden dimension (for 3B model)

# Derived constants
PATCH_FACTOR = VISION_PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28 pixels per merged patch
TOKEN_TO_PIXEL_SIZE = PATCH_FACTOR  # Each vision token = 28x28 pixel region

# =============================================================================
# Attention Extraction Configuration
# =============================================================================

# Which layers to extract attention from
EXTRACT_ALL_LAYERS = True  # Set to False to extract specific layers only
SPECIFIC_LAYERS = [20, 21, 22, 23, 24]  # Last 5 layers (if EXTRACT_ALL_LAYERS=False)

# Attention storage
STORE_ON_CPU = True  # Offload attention to CPU to save GPU memory
USE_FLOAT16 = False  # Use float32 for attention storage (full precision)

# =============================================================================
# Visualization Configuration
# =============================================================================

# Heatmap settings
HEATMAP_COLORMAP = "jet"  # Options: 'jet', 'hot', 'viridis', 'plasma', 'coolwarm'
HEATMAP_ALPHA = 0.5  # Transparency of heatmap overlay (0-1)
HEATMAP_INTERPOLATION = "bilinear"  # Options: 'nearest', 'bilinear', 'bicubic'

# Default aggregation method
DEFAULT_AGGREGATION = "mean"  # Options: 'mean', 'max', 'min'

# Figure settings
FIGURE_DPI = 150
FIGURE_SIZE = (10, 8)

# =============================================================================
# Gradio Interface Configuration
# =============================================================================

# Server settings
GRADIO_SERVER_NAME = "127.0.0.1"  # Local only
GRADIO_SERVER_PORT = 7861
GRADIO_SHARE = False  # Don't create public link

# Interface settings
MAX_IMAGE_SIZE = 1024  # Maximum dimension for uploaded images
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.7

# Example prompts
EXAMPLE_PROMPTS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is the main subject of this image?",
    "Describe the colors and composition of this image.",
    "What is happening in this image?",
]

# =============================================================================
# Processing Configuration
# =============================================================================

# Image processing
IMAGE_MIN_PIXELS = 4 * (PATCH_FACTOR ** 2)  # Minimum pixels for image
IMAGE_MAX_PIXELS = 16384 * (PATCH_FACTOR ** 2)  # Maximum pixels for image

# Memory management
MAX_ATTENTION_CACHE_SIZE = 100  # Maximum number of attention maps to cache
CLEAR_CACHE_AFTER_GENERATION = False  # Clear attention cache after each generation

# =============================================================================
# Paths
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent

# Cache directories
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Debug Settings
# =============================================================================

DEBUG_MODE = False
VERBOSE = True
LOG_ATTENTION_SHAPES = False  # Log attention tensor shapes (useful for debugging)

# =============================================================================
# Advanced Settings
# =============================================================================

# Attention rollout (experimental)
USE_ATTENTION_ROLLOUT = False  # Combine attention across layers using rollout
ROLLOUT_START_LAYER = 0

# Gradient-based attention (experimental)
USE_GRADIENT_ATTENTION = False  # Use gradients to compute attention (requires backward pass)
