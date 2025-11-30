"""
Attention Processing Module for Qwen2.5-VL

This module handles the processing of attention weights to map them
to specific image regions and tokens.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import config
import utils


class AttentionProcessor:
    """
    Process attention weights and map them to image regions.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        original_image_size: Optional[Tuple[int, int]] = None,
        resized_image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize attention processor.

        Args:
            input_ids: Input token IDs tensor (batch_size, seq_len)
            image_grid_thw: Image grid information (num_images, 3) [T, H, W]
            video_grid_thw: Video grid information (num_videos, 3) [T, H, W]
            original_image_size: (width, height) of original image
            resized_image_size: (width, height) of image after preprocessing
        """
        self.input_ids = input_ids
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw
        self.original_image_size = original_image_size
        self.resized_image_size = resized_image_size

        # Find vision token positions
        self.image_token_indices, self.video_token_indices = utils.find_vision_token_indices(input_ids)

        # Calculate vision token ranges
        self.vision_token_ranges = self._calculate_vision_token_ranges()

        # Get patch mapping
        if image_grid_thw is not None and len(self.image_token_indices) > 0:
            self.patch_mapping = self._create_patch_mapping()
        else:
            self.patch_mapping = None
        
        # Store prompt text token info (will be populated later)
        self.prompt_text_tokens = None

    def _calculate_vision_token_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        Calculate the ranges where actual vision embeddings appear in the sequence.

        Returns:
            Dictionary with 'image' and 'video' ranges
        """
        ranges = {'image': [], 'video': []}

        # For images
        if self.image_grid_thw is not None and len(self.image_token_indices) > 0:
            current_pos = self.image_token_indices[0]
            for i, grid_thw in enumerate(self.image_grid_thw):
                t, h, w = grid_thw
                num_tokens = (t * h * w) // (config.SPATIAL_MERGE_SIZE ** 2)
                num_tokens = num_tokens.item()

                # The vision tokens replace the placeholder token
                # So the range starts at the placeholder position
                start = current_pos
                end = current_pos + num_tokens
                ranges['image'].append((start, end))

                # Move to next image if exists
                if i + 1 < len(self.image_token_indices):
                    current_pos = self.image_token_indices[i + 1]

        # For videos (similar logic)
        if self.video_grid_thw is not None and len(self.video_token_indices) > 0:
            current_pos = self.video_token_indices[0]
            for i, grid_thw in enumerate(self.video_grid_thw):
                t, h, w = grid_thw
                num_tokens = (t * h * w) // (config.SPATIAL_MERGE_SIZE ** 2)
                num_tokens = num_tokens.item()

                start = current_pos
                end = current_pos + num_tokens
                ranges['video'].append((start, end))

                if i + 1 < len(self.video_token_indices):
                    current_pos = self.video_token_indices[i + 1]

        return ranges

    def _create_patch_mapping(self) -> Dict[int, Tuple[int, int]]:
        """
        Create mapping from vision token index to patch position.

        Returns:
            Dictionary mapping token index to (row, col) position
        """
        mapping = {}

        # Process each image
        for img_idx, (start, end) in enumerate(self.vision_token_ranges['image']):
            if img_idx >= len(self.image_grid_thw):
                break

            grid_h, grid_w, positions = utils.map_vision_tokens_to_patches(
                self.image_grid_thw,
                img_idx
            )

            # Map each token index to its patch position
            for token_offset, (row, col) in enumerate(positions):
                token_idx = start + token_offset
                if token_idx < end:
                    mapping[token_idx] = (row, col)

        return mapping

    def get_attention_to_vision_tokens(
        self,
        attention_weights: Dict[int, Dict[int, torch.Tensor]],
        token_position: int,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        aggregation_method: str = "mean"
    ) -> Optional[torch.Tensor]:
        """
        Get aggregated attention from a specific token to all vision tokens.

        Args:
            attention_weights: Full attention weights dict
            token_position: Position of the querying token
            layer_indices: Specific layers to use
            head_indices: Specific heads to use
            aggregation_method: How to aggregate attention

        Returns:
            1D tensor of attention values for vision tokens, or None
        """
        print(f"[get_attention_to_vision_tokens] Token position: {token_position}")
        print(f"[get_attention_to_vision_tokens] Image token ranges: {self.vision_token_ranges['image']}")
        
        if not self.vision_token_ranges['image']:
            print("[get_attention_to_vision_tokens] ERROR: No image token ranges found")
            return None

        # Aggregate attention across layers and heads
        print(f"[get_attention_to_vision_tokens] Aggregating attention with method: {aggregation_method}")
        aggregated_attn = utils.aggregate_attention(
            attention_weights,
            layer_indices=layer_indices,
            head_indices=head_indices,
            aggregation_method=aggregation_method
        )
        
        print(f"[get_attention_to_vision_tokens] Aggregated attention shape: {aggregated_attn.shape}")

        # Extract attention from token_position to all positions
        # In autoregressive generation, attention may be (1, seq_len) for a single token
        # or (seq_len, seq_len) for full sequence
        if aggregated_attn.dim() == 2:
            if aggregated_attn.size(0) == 1:
                # Single token attention: shape is (1, seq_len)
                # This is the attention from the current token to all previous tokens
                print(f"[get_attention_to_vision_tokens] Single token attention detected")
                attention_from_token = aggregated_attn[0, :]
            elif token_position < aggregated_attn.size(0):
                # Full attention matrix: shape is (seq_len, seq_len)
                print(f"[get_attention_to_vision_tokens] Full attention matrix detected")
                attention_from_token = aggregated_attn[token_position, :]
            else:
                print(f"[get_attention_to_vision_tokens] ERROR: token_position {token_position} >= seq_len {aggregated_attn.size(0)}")
                return None
        else:
            print(f"[get_attention_to_vision_tokens] ERROR: Unexpected attention shape: {aggregated_attn.shape}")
            return None
            
        print(f"[get_attention_to_vision_tokens] Attention from token shape: {attention_from_token.shape}")

        # Extract attention to vision tokens only
        vision_attention_list = []
        for start, end in self.vision_token_ranges['image']:
            print(f"[get_attention_to_vision_tokens] Extracting vision tokens from {start} to {end}")
            vision_attention_list.append(attention_from_token[start:end])

        if vision_attention_list:
            vision_attention = torch.cat(vision_attention_list)
            print(f"[get_attention_to_vision_tokens] Final vision attention shape: {vision_attention.shape}")
            return vision_attention

        print("[get_attention_to_vision_tokens] ERROR: No vision attention extracted")
        return None

    def create_attention_map(
        self,
        vision_attention: torch.Tensor,
        image_idx: int = 0,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Create a 2D attention map for visualization.

        Args:
            vision_attention: 1D tensor of attention values for vision tokens
            image_idx: Index of the image
            normalize: Whether to normalize attention values

        Returns:
            2D numpy array representing attention map, or None
        """
        print(f"[create_attention_map] Creating map for image_idx: {image_idx}")
        print(f"[create_attention_map] Patch mapping exists: {self.patch_mapping is not None}")
        print(f"[create_attention_map] Image grid thw exists: {self.image_grid_thw is not None}")
        
        if self.patch_mapping is None or self.image_grid_thw is None:
            print("[create_attention_map] ERROR: patch_mapping or image_grid_thw is None")
            return None

        if image_idx >= len(self.image_grid_thw):
            print(f"[create_attention_map] ERROR: image_idx {image_idx} >= num_images {len(self.image_grid_thw)}")
            return None

        # Get grid dimensions
        t, h, w = self.image_grid_thw[image_idx]
        grid_h = h.item() // config.SPATIAL_MERGE_SIZE
        grid_w = w.item() // config.SPATIAL_MERGE_SIZE
        print(f"[create_attention_map] Grid dimensions: t={t}, h={h}, w={w}, grid_h={grid_h}, grid_w={grid_w}")

        # For videos/temporal, we'll average across time
        # For simplicity, we'll create a 2D map for the first frame
        attention_map = np.zeros((grid_h, grid_w), dtype=np.float32)

        # Get token range for this image
        if image_idx >= len(self.vision_token_ranges['image']):
            print(f"[create_attention_map] ERROR: image_idx {image_idx} >= num_image_ranges {len(self.vision_token_ranges['image'])}")
            return None

        start, end = self.vision_token_ranges['image'][image_idx]
        num_tokens = end - start
        print(f"[create_attention_map] Vision token range: {start} to {end} ({num_tokens} tokens)")
        print(f"[create_attention_map] Vision attention length: {len(vision_attention)}")

        # Fill in attention values
        attention_values = vision_attention[:num_tokens].cpu().numpy()

        token_count = 0
        for token_idx in range(start, end):
            if token_idx in self.patch_mapping:
                row, col = self.patch_mapping[token_idx]
                if row < grid_h and col < grid_w and token_count < len(attention_values):
                    attention_map[row, col] = attention_values[token_count]
                    token_count += 1
        
        print(f"[create_attention_map] Filled {token_count} tokens into attention map")

        # Normalize if requested
        if normalize:
            min_val = attention_map.min()
            max_val = attention_map.max()
            print(f"[create_attention_map] Attention range before norm: [{min_val}, {max_val}]")
            if max_val - min_val > 1e-8:
                attention_map = (attention_map - min_val) / (max_val - min_val)
                print(f"[create_attention_map] Normalized attention map")
            else:
                print(f"[create_attention_map] WARNING: Attention values are constant, skipping normalization")

        print(f"[create_attention_map] Final attention map shape: {attention_map.shape}")
        return attention_map

    def get_attention_heatmap_for_token(
        self,
        attention_weights: Dict[int, Dict[int, torch.Tensor]],
        token_position: int,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        aggregation_method: str = "mean",
        image_idx: int = 0,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get a complete attention heatmap for a specific token.

        This is the main method that combines all steps:
        1. Extract attention to vision tokens
        2. Create 2D attention map
        3. Return as numpy array ready for visualization

        Args:
            attention_weights: Full attention weights dict
            token_position: Position of the querying token
            layer_indices: Specific layers to use
            head_indices: Specific heads to use
            aggregation_method: How to aggregate attention
            image_idx: Index of the image
            normalize: Whether to normalize attention values

        Returns:
            2D numpy array representing attention heatmap
        """
        print(f"[AttentionProcessor] Getting attention heatmap for token at position {token_position}")
        print(f"[AttentionProcessor] Vision token ranges: {self.vision_token_ranges}")
        print(f"[AttentionProcessor] Patch mapping exists: {self.patch_mapping is not None}")
        if self.patch_mapping:
            print(f"[AttentionProcessor] Patch mapping size: {len(self.patch_mapping)}")
        
        # Get attention to vision tokens
        vision_attention = self.get_attention_to_vision_tokens(
            attention_weights,
            token_position,
            layer_indices,
            head_indices,
            aggregation_method
        )

        if vision_attention is None:
            print("[AttentionProcessor] ERROR: vision_attention is None")
            return None
        
        print(f"[AttentionProcessor] Vision attention shape: {vision_attention.shape}")

        # Create 2D attention map
        attention_map = self.create_attention_map(
            vision_attention,
            image_idx,
            normalize
        )

        if attention_map is None:
            print("[AttentionProcessor] ERROR: attention_map is None")
            return None
        
        print(f"[AttentionProcessor] Created attention map with shape: {attention_map.shape}")

        return attention_map

    def get_pixel_coordinates_for_attention(
        self,
        image_idx: int = 0
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Get pixel coordinates for each attention map cell.

        Args:
            image_idx: Index of the image

        Returns:
            List of (x1, y1, x2, y2) bounding boxes in original image coordinates
        """
        if self.patch_mapping is None or self.image_grid_thw is None:
            return None

        if image_idx >= len(self.vision_token_ranges['image']):
            return None

        # Get positions for this image
        start, end = self.vision_token_ranges['image'][image_idx]
        positions = []

        for token_idx in range(start, end):
            if token_idx in self.patch_mapping:
                positions.append(self.patch_mapping[token_idx])

        # Convert to pixel coordinates if we have size information
        if self.original_image_size and self.resized_image_size:
            pixel_coords = utils.patch_positions_to_pixel_coords(
                positions,
                self.original_image_size,
                self.resized_image_size
            )
            return pixel_coords

        return positions

    def get_attention_distribution(
        self,
        attention_weights: Dict[int, Dict[int, torch.Tensor]],
        token_position: int,
        input_length: int,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        aggregation_method: str = "mean"
    ) -> Optional[Dict[str, float]]:
        """
        Calculate attention distribution between image and prompt text tokens.
        
        This is a statistical measure that should remain consistent regardless of
        visualization preferences. Typically called with aggregation_method="mean"
        to ensure objective, reproducible results.
        
        Args:
            attention_weights: Attention weights {layer_idx: {head_idx: tensor}}
            token_position: Position of token to analyze
            input_length: Length of input sequence (prompt + image tokens, excluding generated tokens)
            layer_indices: Layers to aggregate
            head_indices: Heads to aggregate
            aggregation_method: How to aggregate (typically "mean" for consistent statistics)
            
        Returns:
            Dictionary with 'image_percentage' and 'prompt_text_percentage'
        """
        # Aggregate attention across layers and heads using utils
        aggregated_attn = utils.aggregate_attention(
            attention_weights,
            layer_indices=layer_indices,
            head_indices=head_indices,
            aggregation_method=aggregation_method
        )
        
        if aggregated_attn is None:
            return None
        
        # Extract attention from the specified token to all positions
        if aggregated_attn.dim() == 2:
            if aggregated_attn.size(0) == 1:
                attention_from_token = aggregated_attn[0, :]
            elif token_position < aggregated_attn.size(0):
                attention_from_token = aggregated_attn[token_position, :]
            else:
                return None
        else:
            return None
        
        # Calculate total attention to vision tokens (image)
        vision_attention_sum = 0.0
        vision_token_positions = set()
        vision_token_count = 0
        for start, end in self.vision_token_ranges['image']:
            vision_attention_sum += attention_from_token[start:end].sum().item()
            vision_token_positions.update(range(start, end))
            vision_token_count += (end - start)
        for start, end in self.vision_token_ranges['video']:
            vision_attention_sum += attention_from_token[start:end].sum().item()
            vision_token_positions.update(range(start, end))
            vision_token_count += (end - start)
        
        # Calculate attention to prompt text tokens (input tokens that are not vision tokens)
        # Only consider tokens within input_length (exclude generated tokens)
        prompt_text_attention_sum = 0.0
        prompt_text_token_count = 0
        for pos in range(min(input_length, len(attention_from_token))):
            if pos not in vision_token_positions:
                prompt_text_attention_sum += attention_from_token[pos].item()
                prompt_text_token_count += 1
        
        # Calculate total attention within input range
        total_attention_in_input = vision_attention_sum + prompt_text_attention_sum
        
        # Calculate average attention per token
        avg_vision_attention = vision_attention_sum / vision_token_count if vision_token_count > 0 else 0.0
        avg_text_attention = prompt_text_attention_sum / prompt_text_token_count if prompt_text_token_count > 0 else 0.0
        
        # Calculate percentages
        if total_attention_in_input > 0:
            image_percentage = (vision_attention_sum / total_attention_in_input) * 100
            prompt_text_percentage = (prompt_text_attention_sum / total_attention_in_input) * 100
        else:
            image_percentage = 0.0
            prompt_text_percentage = 0.0
        
        # Debug logging
        print(f"[get_attention_distribution] Token counts: Vision={vision_token_count}, Prompt Text={prompt_text_token_count}")
        print(f"[get_attention_distribution] Attention sums: Vision={vision_attention_sum:.4f}, Prompt Text={prompt_text_attention_sum:.4f}")
        print(f"[get_attention_distribution] Average attention per token: Vision={avg_vision_attention:.6f}, Prompt Text={avg_text_attention:.6f}")
        print(f"[get_attention_distribution] Percentages: Image={image_percentage:.1f}%, Prompt Text={prompt_text_percentage:.1f}%")
        
        return {
            'image_percentage': image_percentage,
            'prompt_text_percentage': prompt_text_percentage,
            'total_attention_in_input': total_attention_in_input,
            'vision_token_count': vision_token_count,
            'prompt_text_token_count': prompt_text_token_count,
            'avg_vision_attention': avg_vision_attention,
            'avg_text_attention': avg_text_attention
        }

    def set_prompt_text_tokens(self, tokenizer):
        """
        Extract and store prompt text token positions and their text.
        
        Args:
            tokenizer: Tokenizer instance for decoding
        """
        self.prompt_text_tokens = utils.get_prompt_text_token_positions(
            self.input_ids,
            tokenizer,
            self.vision_token_ranges
        )
        
        if config.VERBOSE:
            print(f"[AttentionProcessor] Found {len(self.prompt_text_tokens)} prompt text tokens")
    
    def get_prompt_text_token_info(self) -> Optional[List[Tuple[int, str]]]:
        """
        Get prompt text token positions and their decoded text.
        
        Returns:
            List of (position, token_text) tuples, or None if not yet extracted
        """
        return self.prompt_text_tokens
    
    def get_prompt_token_attention_heatmap(
        self,
        attention_weights: Dict[int, Dict[int, Dict[int, torch.Tensor]]],
        prompt_token_position: int,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        aggregation_method: str = "mean",
        image_idx: int = 0,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get attention heatmap for a prompt text token to image regions.
        
        This uses the prefill attention (first generation step) where prompt tokens
        attend to all input tokens including vision tokens.
        
        Args:
            attention_weights: Full attention weights dict (step-based format)
                {step_key: {layer_idx: {head_idx: attention_tensor}}}
            prompt_token_position: Position of the prompt token in input sequence
            layer_indices: Specific layers to use
            head_indices: Specific heads to use
            aggregation_method: How to aggregate attention
            image_idx: Index of the image
            normalize: Whether to normalize attention values
            
        Returns:
            2D numpy array representing attention heatmap
        """
        print(f"[get_prompt_token_attention_heatmap] Processing prompt token at position {prompt_token_position}")
        
        # Get the prefill step (first generation step with full input sequence)
        # The step key is the input length (full input sequence)
        input_length = self.input_ids.shape[1] if self.input_ids.dim() == 2 else len(self.input_ids)
        prefill_step_key = input_length
        
        print(f"[get_prompt_token_attention_heatmap] Input length: {input_length}")
        print(f"[get_prompt_token_attention_heatmap] Looking for prefill step key: {prefill_step_key}")
        print(f"[get_prompt_token_attention_heatmap] Available step keys: {sorted(attention_weights.keys())}")
        
        # Get attention for the prefill step
        if prefill_step_key not in attention_weights:
            print(f"[get_prompt_token_attention_heatmap] WARNING: Prefill step {prefill_step_key} not found")
            # Try to find the closest step (should be the first/smallest step)
            available_steps = sorted(attention_weights.keys())
            if available_steps:
                prefill_step_key = available_steps[0]
                print(f"[get_prompt_token_attention_heatmap] Using first available step: {prefill_step_key}")
            else:
                print(f"[get_prompt_token_attention_heatmap] ERROR: No attention data available")
                return None
        
        step_attention = attention_weights[prefill_step_key]
        
        # Get attention heatmap using existing method
        attention_map = self.get_attention_heatmap_for_token(
            step_attention,
            token_position=prompt_token_position,
            layer_indices=layer_indices,
            head_indices=head_indices,
            aggregation_method=aggregation_method,
            image_idx=image_idx,
            normalize=normalize
        )
        
        if attention_map is None:
            print(f"[get_prompt_token_attention_heatmap] ERROR: Failed to create attention map")
            return None
        
        print(f"[get_prompt_token_attention_heatmap] Successfully created attention map with shape: {attention_map.shape}")
        return attention_map

    def get_sequence_info(self) -> Dict[str, Any]:
        """
        Get information about the processed sequence.

        Returns:
            Dictionary with sequence information
        """
        total_vision_tokens = sum(
            end - start for start, end in self.vision_token_ranges['image']
        )
        total_vision_tokens += sum(
            end - start for start, end in self.vision_token_ranges['video']
        )

        return {
            'total_tokens': self.input_ids.size(-1),
            'num_images': len(self.vision_token_ranges['image']),
            'num_videos': len(self.vision_token_ranges['video']),
            'total_vision_tokens': total_vision_tokens,
            'image_token_indices': self.image_token_indices,
            'video_token_indices': self.video_token_indices,
            'vision_token_ranges': self.vision_token_ranges
        }

    def get_sliding_window_attention_maps(
        self,
        attention_weights: Dict[int, Dict[int, Dict[int, torch.Tensor]]],
        generated_token_positions: List[int],
        window_size: int = 3,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        aggregation_method: str = "mean",
        image_idx: int = 0,
        normalize: bool = True
    ) -> List[Tuple[List[int], np.ndarray]]:
        """
        使用滑动窗口计算生成token的注意力热力图。
        
        对于生成的token序列，使用长度为window_size的滑动窗口，
        每个窗口内的token注意力进行平均，得到一个热力图。
        
        Args:
            attention_weights: 完整的注意力权重字典
                可以是两种格式：
                1. {layer_idx: {head_idx: tensor}} - 旧格式
                2. {step_key: {layer_idx: {head_idx: tensor}}} - 新格式（按生成步骤）
            generated_token_positions: 生成的token位置列表
            window_size: 滑动窗口大小（默认3）
            layer_indices: 要使用的层索引
            head_indices: 要使用的头索引
            aggregation_method: 聚合方法
            image_idx: 图像索引
            normalize: 是否归一化
            
        Returns:
            List of (window_positions, attention_map) tuples
            每个元组包含窗口内的token位置列表和对应的注意力热力图
        """
        print(f"[get_sliding_window_attention_maps] Processing {len(generated_token_positions)} generated tokens")
        print(f"[get_sliding_window_attention_maps] Window size: {window_size}")
        
        if len(generated_token_positions) < window_size:
            print(f"[get_sliding_window_attention_maps] WARNING: Not enough tokens ({len(generated_token_positions)}) for window size {window_size}")
            window_size = len(generated_token_positions)
        
        # 检测attention_weights的格式
        is_step_based = False
        if attention_weights:
            first_key = next(iter(attention_weights))
            first_value = attention_weights[first_key]
            # 如果第一层的值还是dict，说明是step-based格式
            if isinstance(first_value, dict) and first_value:
                second_key = next(iter(first_value))
                if isinstance(first_value[second_key], dict):
                    is_step_based = True
        
        print(f"[get_sliding_window_attention_maps] Attention format: {'step-based' if is_step_based else 'layer-based'}")
        
        results = []
        
        # 滑动窗口遍历生成的tokens
        for i in range(len(generated_token_positions) - window_size + 1):
            window_positions = generated_token_positions[i:i+window_size]
            print(f"\n[get_sliding_window_attention_maps] Window {i}: positions {window_positions}")
            
            # 收集窗口内所有token的注意力
            window_vision_attentions = []
            
            for token_pos in window_positions:
                # 根据格式获取该token的注意力
                if is_step_based:
                    # 新格式：需要获取对应步骤的注意力
                    step_key = token_pos
                    if step_key in attention_weights:
                        step_attention = attention_weights[step_key]
                    else:
                        # 尝试找到最近的步骤
                        available_steps = sorted(attention_weights.keys())
                        if available_steps:
                            step_key = min(available_steps, key=lambda x: abs(x - token_pos))
                            step_attention = attention_weights[step_key]
                            print(f"[get_sliding_window_attention_maps] Using closest step {step_key} for token {token_pos}")
                        else:
                            continue
                else:
                    # 旧格式：直接使用
                    step_attention = attention_weights
                
                # 获取该token对视觉token的注意力
                vision_attention = self.get_attention_to_vision_tokens(
                    step_attention,
                    token_pos,
                    layer_indices,
                    head_indices,
                    aggregation_method
                )
                
                if vision_attention is not None:
                    window_vision_attentions.append(vision_attention)
            
            if not window_vision_attentions:
                print(f"[get_sliding_window_attention_maps] WARNING: No attention found for window {i}")
                continue
            
            # 对窗口内的注意力求平均
            avg_vision_attention = torch.stack(window_vision_attentions).mean(dim=0)
            print(f"[get_sliding_window_attention_maps] Averaged {len(window_vision_attentions)} attentions, shape: {avg_vision_attention.shape}")
            
            # 创建2D注意力热力图
            attention_map = self.create_attention_map(
                avg_vision_attention,
                image_idx,
                normalize
            )
            
            if attention_map is not None:
                results.append((window_positions, attention_map))
                print(f"[get_sliding_window_attention_maps] Created attention map for window {i}, shape: {attention_map.shape}")
            else:
                print(f"[get_sliding_window_attention_maps] WARNING: Failed to create attention map for window {i}")
        
        print(f"\n[get_sliding_window_attention_maps] Total windows processed: {len(results)}")
        return results


def create_attention_processor(
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    original_image: Optional[Image.Image] = None,
    resized_image_size: Optional[Tuple[int, int]] = None
) -> AttentionProcessor:
    """
    Convenience function to create an AttentionProcessor.

    Args:
        input_ids: Input token IDs
        image_grid_thw: Image grid information
        original_image: Original PIL image
        resized_image_size: Size of resized image used by model

    Returns:
        AttentionProcessor instance
    """
    original_size = None
    if original_image:
        original_size = (original_image.width, original_image.height)

    return AttentionProcessor(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        original_image_size=original_size,
        resized_image_size=resized_image_size
    )
