"""
Utility functions for Qwen2.5-VL Attention Visualization
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import config


def find_vision_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int = config.IMAGE_TOKEN_ID,
    video_token_id: int = config.VIDEO_TOKEN_ID
) -> Tuple[List[int], List[int]]:
    """
    Find the positions of vision tokens (image/video) in the input sequence.

    Args:
        input_ids: Input token IDs tensor of shape (batch_size, seq_len)
        image_token_id: Token ID for images
        video_token_id: Token ID for videos

    Returns:
        Tuple of (image_token_indices, video_token_indices)
    """
    # Flatten if batch dimension exists
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
    video_indices = (input_ids == video_token_id).nonzero(as_tuple=True)[0].tolist()

    return image_indices, video_indices


def get_vision_token_count(image_grid_thw: torch.Tensor) -> int:
    """
    Calculate the number of vision tokens from grid_thw.

    Args:
        image_grid_thw: Tensor of shape (num_images, 3) containing [T, H, W] for each image

    Returns:
        Total number of vision tokens
    """
    if image_grid_thw is None:
        return 0

    # Each element in grid_thw is [temporal, height, width]
    # After spatial merging, each spatial_merge_size x spatial_merge_size patch becomes one token
    spatial_merge_size_sq = config.SPATIAL_MERGE_SIZE ** 2

    total_tokens = 0
    for thw in image_grid_thw:
        t, h, w = thw
        # Number of tokens = (T * H * W) / (spatial_merge_size^2)
        tokens_per_image = (t * h * w) // spatial_merge_size_sq
        total_tokens += tokens_per_image

    return total_tokens


def map_vision_tokens_to_patches(
    image_grid_thw: torch.Tensor,
    image_idx: int = 0
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Map vision tokens back to their corresponding patch positions in the image.

    Args:
        image_grid_thw: Tensor of shape (num_images, 3) containing [T, H, W]
        image_idx: Index of the image to process

    Returns:
        Tuple of (grid_height, grid_width, list of (row, col) positions)
    """
    if image_idx >= len(image_grid_thw):
        raise ValueError(f"image_idx {image_idx} out of range")

    t, h, w = image_grid_thw[image_idx]
    t, h, w = t.item(), h.item(), w.item()

    # After spatial merging
    grid_h = h // config.SPATIAL_MERGE_SIZE
    grid_w = w // config.SPATIAL_MERGE_SIZE

    # Generate (row, col) positions for each token
    positions = []
    for temporal in range(t):
        for row in range(grid_h):
            for col in range(grid_w):
                positions.append((row, col))

    return grid_h, grid_w, positions


def patch_positions_to_pixel_coords(
    patch_positions: List[Tuple[int, int]],
    original_image_size: Tuple[int, int],
    resized_image_size: Tuple[int, int]
) -> List[Tuple[int, int, int, int]]:
    """
    Convert patch positions to pixel coordinates (bounding boxes).

    Args:
        patch_positions: List of (row, col) patch positions
        original_image_size: (width, height) of original image
        resized_image_size: (width, height) of resized image used by model

    Returns:
        List of (x1, y1, x2, y2) bounding boxes in original image coordinates
    """
    orig_w, orig_h = original_image_size
    resized_w, resized_h = resized_image_size

    # Scale factor to map from resized to original
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    pixel_coords = []
    for row, col in patch_positions:
        # Coordinates in resized image
        x1_resized = col * config.TOKEN_TO_PIXEL_SIZE
        y1_resized = row * config.TOKEN_TO_PIXEL_SIZE
        x2_resized = x1_resized + config.TOKEN_TO_PIXEL_SIZE
        y2_resized = y1_resized + config.TOKEN_TO_PIXEL_SIZE

        # Scale to original image
        x1 = int(x1_resized * scale_x)
        y1 = int(y1_resized * scale_y)
        x2 = int(x2_resized * scale_x)
        y2 = int(y2_resized * scale_y)

        # Clamp to image boundaries
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        pixel_coords.append((x1, y1, x2, y2))

    return pixel_coords


def aggregate_attention(
    attention_weights: Dict[int, Dict[int, torch.Tensor]],
    layer_indices: Optional[List[int]] = None,
    head_indices: Optional[List[int]] = None,
    aggregation_method: str = "mean"
) -> torch.Tensor:
    """
    Aggregate attention weights across layers and heads.

    Args:
        attention_weights: Dict of {layer_idx: {head_idx: attention_tensor}}
        layer_indices: Specific layers to aggregate (None = all layers)
        head_indices: Specific heads to aggregate (None = all heads)
        aggregation_method: How to aggregate ('mean', 'max', 'min')

    Returns:
        Aggregated attention tensor of shape (seq_len, seq_len)
    """
    selected_attentions = []

    for layer_idx, layer_attn in attention_weights.items():
        # Skip if specific layers requested and this isn't one
        if layer_indices is not None and layer_idx not in layer_indices:
            continue

        for head_idx, attn in layer_attn.items():
            # Skip if specific heads requested and this isn't one
            if head_indices is not None and head_idx not in head_indices:
                continue

            # Attention shape: (batch, num_heads, seq_len, seq_len) or (seq_len, seq_len)
            if attn.dim() == 4:
                attn = attn[0, head_idx]  # Extract batch 0, specific head
            elif attn.dim() == 3:
                attn = attn[head_idx]  # Extract specific head

            selected_attentions.append(attn)

    if not selected_attentions:
        raise ValueError("No attention weights found with specified layer/head indices")

    # Stack all selected attentions
    stacked = torch.stack(selected_attentions, dim=0)  # (num_selections, seq_len, seq_len)

    # Aggregate
    if aggregation_method == "mean":
        result = stacked.mean(dim=0)
    elif aggregation_method == "max":
        result = stacked.max(dim=0)[0]
    elif aggregation_method == "min":
        result = stacked.min(dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return result


def normalize_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention values to [0, 1] range.

    Args:
        attention: Attention tensor

    Returns:
        Normalized attention tensor
    """
    min_val = attention.min()
    max_val = attention.max()

    if max_val - min_val < 1e-8:
        return torch.zeros_like(attention)

    return (attention - min_val) / (max_val - min_val)


def resize_image_for_display(
    image: Image.Image,
    max_size: int = config.MAX_IMAGE_SIZE
) -> Image.Image:
    """
    Resize image for display while maintaining aspect ratio.

    Args:
        image: PIL Image
        max_size: Maximum dimension size

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def decode_tokens_to_text(
    token_ids: List[int],
    tokenizer
) -> List[str]:
    """
    Decode individual token IDs to text strings.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        List of decoded token strings
    """
    tokens = []
    for token_id in token_ids:
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        tokens.append(token_str)

    return tokens


def get_layer_head_counts(model) -> Tuple[int, int]:
    """
    Get the number of layers and attention heads in the model.

    Args:
        model: Qwen2.5-VL model

    Returns:
        Tuple of (num_layers, num_heads)
    """
    config = model.config.text_config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    return num_layers, num_heads


def create_attention_mask_visualization(
    attention: torch.Tensor,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Create a binary mask from attention values above threshold.

    Args:
        attention: Attention tensor
        threshold: Threshold for creating binary mask

    Returns:
        Binary mask as numpy array
    """
    attention_np = attention.cpu().numpy() if isinstance(attention, torch.Tensor) else attention
    mask = (attention_np > threshold).astype(np.uint8)

    return mask


def get_prompt_text_token_positions(
    input_ids: torch.Tensor,
    tokenizer,
    vision_token_ranges: Dict[str, List[Tuple[int, int]]]
) -> List[Tuple[int, str]]:
    """
    Extract prompt text token positions and their text, excluding vision and special tokens.
    
    Args:
        input_ids: Input token IDs tensor (batch_size, seq_len) or (seq_len,)
        tokenizer: Tokenizer instance
        vision_token_ranges: Dictionary with 'image' and 'video' token ranges
        
    Returns:
        List of (position, token_text) tuples for prompt text tokens
    """
    # Flatten if batch dimension exists
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    
    # Build set of vision token positions
    vision_positions = set()
    for start, end in vision_token_ranges.get('image', []):
        vision_positions.update(range(start, end))
    for start, end in vision_token_ranges.get('video', []):
        vision_positions.update(range(start, end))
    
    # Get special token IDs to exclude
    special_ids = {
        config.VISION_START_TOKEN_ID,
        config.VISION_END_TOKEN_ID,
        config.IMAGE_TOKEN_ID,
        config.VIDEO_TOKEN_ID,
    }
    
    # Add tokenizer special tokens if available
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    
    # Extract prompt text tokens
    prompt_positions = []
    for pos in range(len(input_ids)):
        token_id = input_ids[pos].item()
        if pos not in vision_positions and token_id not in special_ids:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            prompt_positions.append((pos, token_text))
    
    return prompt_positions


def log_attention_info(
    attention_weights: Dict[int, Dict[int, torch.Tensor]],
    verbose: bool = True
):
    """
    Log information about extracted attention weights.

    Args:
        attention_weights: Dict of attention weights
        verbose: Whether to print detailed information
    """
    if not verbose:
        return

    num_layers = len(attention_weights)
    print(f"\n{'='*60}")
    print(f"Attention Extraction Summary")
    print(f"{'='*60}")
    print(f"Number of layers: {num_layers}")

    for layer_idx in sorted(attention_weights.keys())[:3]:  # Show first 3 layers
        num_heads = len(attention_weights[layer_idx])
        first_head_shape = attention_weights[layer_idx][0].shape
        print(f"Layer {layer_idx}: {num_heads} heads, shape: {first_head_shape}")

    if num_layers > 3:
        print(f"... ({num_layers - 3} more layers)")

    print(f"{'='*60}\n")
