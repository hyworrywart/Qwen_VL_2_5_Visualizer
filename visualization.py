"""
Visualization Module for Qwen2.5-VL Attention

This module handles the creation of attention heatmaps and overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Union
import config
import os
from pathlib import Path


class AttentionVisualizer:
    """
    Create attention heatmap visualizations.
    """

    def __init__(
        self,
        colormap: str = config.HEATMAP_COLORMAP,
        alpha: float = config.HEATMAP_ALPHA,
        interpolation: str = config.HEATMAP_INTERPOLATION
    ):
        """
        Initialize visualizer.

        Args:
            colormap: Matplotlib colormap name
            alpha: Transparency of heatmap overlay (0-1)
            interpolation: Interpolation method for upscaling heatmap
        """
        self.colormap = colormap
        self.alpha = alpha
        self.interpolation = interpolation

    def create_heatmap_overlay(
        self,
        image: Union[Image.Image, np.ndarray],
        attention_map: np.ndarray,
        resize_to_image: bool = True,
        return_pil: bool = True
    ) -> Union[Image.Image, np.ndarray]:
        """
        Create a heatmap overlay on an image.

        Args:
            image: Original image (PIL Image or numpy array)
            attention_map: 2D attention map (grid_h, grid_w)
            resize_to_image: Whether to resize heatmap to match image size
            return_pil: Whether to return PIL Image (True) or numpy array (False)

        Returns:
            Image with heatmap overlay
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Ensure image is RGB
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Resize attention map to match image if requested
        if resize_to_image:
            target_size = (img_array.shape[1], img_array.shape[0])  # (width, height)
            interpolation_cv2 = {
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC
            }.get(self.interpolation, cv2.INTER_LINEAR)

            attention_resized = cv2.resize(
                attention_map,
                target_size,
                interpolation=interpolation_cv2
            )
        else:
            attention_resized = attention_map

        # Normalize attention map to [0, 1]
        attention_norm = (attention_resized - attention_resized.min()) / (
            attention_resized.max() - attention_resized.min() + 1e-8
        )

        # Apply colormap
        cmap = cm.get_cmap(self.colormap)
        heatmap_colored = cmap(attention_norm)
        heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

        # Blend with original image
        blended = cv2.addWeighted(
            img_array,
            1 - self.alpha,
            heatmap_rgb,
            self.alpha,
            0
        )

        if return_pil:
            return Image.fromarray(blended)
        return blended

    def create_side_by_side_comparison(
        self,
        image: Union[Image.Image, np.ndarray],
        attention_map: np.ndarray,
        titles: Optional[Tuple[str, str]] = None
    ) -> Image.Image:
        """
        Create a side-by-side comparison of original image and heatmap overlay.

        Args:
            image: Original image
            attention_map: 2D attention map
            titles: Tuple of (original_title, heatmap_title)

        Returns:
            PIL Image with side-by-side comparison
        """
        # Create heatmap overlay
        overlay = self.create_heatmap_overlay(image, attention_map, return_pil=False)

        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(img_array)
        axes[0].axis('off')
        if titles:
            axes[0].set_title(titles[0], fontsize=14)
        else:
            axes[0].set_title('Original Image', fontsize=14)

        # Heatmap overlay
        axes[1].imshow(overlay)
        axes[1].axis('off')
        if titles:
            axes[1].set_title(titles[1], fontsize=14)
        else:
            axes[1].set_title('Attention Heatmap', fontsize=14)

        plt.tight_layout()

        # Convert figure to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        result_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return Image.fromarray(result_img)

    def create_multi_token_comparison(
        self,
        image: Union[Image.Image, np.ndarray],
        attention_maps: List[np.ndarray],
        token_texts: List[str],
        max_cols: int = 3
    ) -> Image.Image:
        """
        Create a grid comparison of attention for multiple tokens.

        Args:
            image: Original image
            attention_maps: List of 2D attention maps
            token_texts: List of token text labels
            max_cols: Maximum columns in grid

        Returns:
            PIL Image with grid comparison
        """
        num_maps = len(attention_maps)
        num_cols = min(max_cols, num_maps)
        num_rows = (num_maps + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

        # Flatten axes for easy iteration
        if num_rows == 1 and num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, (attn_map, token_text) in enumerate(zip(attention_maps, token_texts)):
            overlay = self.create_heatmap_overlay(image, attn_map, return_pil=False)
            axes[idx].imshow(overlay)
            axes[idx].axis('off')
            axes[idx].set_title(f'Token: "{token_text}"', fontsize=10)

        # Hide unused subplots
        for idx in range(num_maps, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        result_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return Image.fromarray(result_img)

    def create_heatmap_only(
        self,
        attention_map: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> Image.Image:
        """
        Create a standalone heatmap visualization.

        Args:
            attention_map: 2D attention map
            figsize: Figure size

        Returns:
            PIL Image of heatmap
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(attention_map, cmap=self.colormap, interpolation=self.interpolation)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)

        plt.tight_layout()

        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        result_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return Image.fromarray(result_img)

    def create_attention_statistics_plot(
        self,
        attention_map: np.ndarray,
        title: str = "Attention Statistics"
    ) -> Image.Image:
        """
        Create a statistical visualization of attention distribution.

        Args:
            attention_map: 2D attention map
            title: Plot title

        Returns:
            PIL Image with statistics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Heatmap
        im = axes[0].imshow(attention_map, cmap=self.colormap)
        axes[0].set_title('Attention Heatmap')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0])

        # Histogram
        axes[1].hist(attention_map.flatten(), bins=50, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution')
        axes[1].grid(True, alpha=0.3)

        # Row/Column sums
        row_sums = attention_map.sum(axis=1)
        col_sums = attention_map.sum(axis=0)

        axes[2].plot(row_sums, label='Row Sums', marker='o')
        axes[2].plot(col_sums, label='Column Sums', marker='s')
        axes[2].set_xlabel('Index')
        axes[2].set_ylabel('Sum')
        axes[2].set_title('Spatial Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        result_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return Image.fromarray(result_img)


def create_visualizer(
    colormap: str = config.HEATMAP_COLORMAP,
    alpha: float = config.HEATMAP_ALPHA
) -> AttentionVisualizer:
    """
    Create an AttentionVisualizer instance.

    Args:
        colormap: Matplotlib colormap name
        alpha: Transparency of overlay

    Returns:
        AttentionVisualizer instance
    """
    return AttentionVisualizer(colormap=colormap, alpha=alpha)


def visualize_attention_on_image(
    image: Union[Image.Image, np.ndarray],
    attention_map: np.ndarray,
    colormap: str = config.HEATMAP_COLORMAP,
    alpha: float = config.HEATMAP_ALPHA,
    show_comparison: bool = False
) -> Image.Image:
    """
    Quick function to visualize attention on an image.

    Args:
        image: Original image
        attention_map: 2D attention map
        colormap: Colormap to use
        alpha: Overlay transparency
        show_comparison: Whether to show side-by-side comparison

    Returns:
        PIL Image with visualization
    """
    visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)

    if show_comparison:
        return visualizer.create_side_by_side_comparison(image, attention_map)
    else:
        return visualizer.create_heatmap_overlay(image, attention_map)


def create_layer_head_visualization(
    image: Union[Image.Image, np.ndarray],
    attention_maps_dict: dict,
    selected_layers: List[int],
    num_heads_to_show: int = 4
) -> Image.Image:
    """
    Create visualization showing attention from multiple layers and heads.

    Args:
        image: Original image
        attention_maps_dict: Dict of {layer_idx: {head_idx: attention_map}}
        selected_layers: List of layer indices to visualize
        num_heads_to_show: Number of heads to show per layer

    Returns:
        PIL Image with multi-layer/head visualization
    """
    visualizer = AttentionVisualizer()

    all_maps = []
    all_labels = []

    for layer_idx in selected_layers:
        if layer_idx not in attention_maps_dict:
            continue

        layer_heads = attention_maps_dict[layer_idx]
        head_indices = sorted(layer_heads.keys())[:num_heads_to_show]

        for head_idx in head_indices:
            attn_map = layer_heads[head_idx]
            all_maps.append(attn_map)
            all_labels.append(f"L{layer_idx}H{head_idx}")

    if not all_maps:
        # Return original image if no attention maps
        if isinstance(image, Image.Image):
            return image
        return Image.fromarray(image)

    return visualizer.create_multi_token_comparison(image, all_maps, all_labels)


def create_sliding_window_attention_video(
    image: Union[Image.Image, np.ndarray],
    sliding_window_results: List[Tuple[List[int], np.ndarray]],
    token_texts: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    fps: int = 2,
    colormap: str = config.HEATMAP_COLORMAP,
    alpha: float = config.HEATMAP_ALPHA,
    show_token_info: bool = True,
    video_size: Optional[Tuple[int, int]] = None
) -> str:
    """
    创建滑动窗口注意力热力图的动态视频。
    
    Args:
        image: 原始图像
        sliding_window_results: 滑动窗口结果列表，每个元素为(window_positions, attention_map)
        token_texts: token文本列表（可选）
        output_path: 输出视频路径，如果为None则自动生成
        fps: 视频帧率（每秒帧数）
        colormap: 热力图颜色映射
        alpha: 热力图透明度
        show_token_info: 是否在视频中显示token信息
        video_size: 视频尺寸(width, height)，如果为None则使用图像尺寸
        
    Returns:
        输出视频文件路径
    """
    print(f"\n[create_sliding_window_attention_video] Creating video with {len(sliding_window_results)} frames")
    
    if len(sliding_window_results) == 0:
        raise ValueError("No sliding window results to create video")
    
    # 转换图像为numpy数组
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # 确保图像是RGB格式
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # 创建可视化器
    visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)
    
    # 确定视频尺寸
    if video_size is None:
        if show_token_info:
            # 为token信息预留空间（在图像上方）
            video_width = img_array.shape[1]
            video_height = img_array.shape[0] + 100  # 额外100像素用于文本
        else:
            video_width = img_array.shape[1]
            video_height = img_array.shape[0]
    else:
        video_width, video_height = video_size
    
    # 设置输出路径
    if output_path is None:
        output_path = config.OUTPUT_DIR / f"sliding_window_attention_{config.OUTPUT_DIR.name}.mp4"
        # 确保文件名唯一
        counter = 1
        while output_path.exists():
            output_path = config.OUTPUT_DIR / f"sliding_window_attention_{counter}.mp4"
            counter += 1
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[create_sliding_window_attention_video] Output path: {output_path}")
    print(f"[create_sliding_window_attention_video] Video size: {video_width}x{video_height}")
    print(f"[create_sliding_window_attention_video] FPS: {fps}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (video_width, video_height)
    )
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to create video writer at {output_path}")
    
    # 为每个窗口创建一帧
    for frame_idx, (window_positions, attention_map) in enumerate(sliding_window_results):
        print(f"[create_sliding_window_attention_video] Processing frame {frame_idx + 1}/{len(sliding_window_results)}")
        
        # 创建热力图叠加
        overlay = visualizer.create_heatmap_overlay(
            img_array,
            attention_map,
            resize_to_image=True,
            return_pil=False
        )
        
        # 如果需要显示token信息，添加文本
        if show_token_info:
            # 创建带文本区域的画布
            canvas = np.ones((video_height, video_width, 3), dtype=np.uint8) * 255
            
            # 将overlay放置在画布下方
            y_offset = 100
            canvas[y_offset:y_offset+overlay.shape[0], :overlay.shape[1]] = overlay
            
            # 添加窗口信息文本
            window_text = f"Window {frame_idx + 1}/{len(sliding_window_results)}"
            positions_text = f"Token positions: {window_positions}"
            
            # 如果有token文本，也显示
            if token_texts:
                tokens_in_window = []
                for pos in window_positions:
                    if pos < len(token_texts):
                        tokens_in_window.append(token_texts[pos])
                tokens_text = f"Tokens: {' '.join(tokens_in_window[:10])}"  # 最多显示10个token
                if len(tokens_in_window) > 10:
                    tokens_text += "..."
            else:
                tokens_text = None
            
            # 绘制文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            font_color = (0, 0, 0)  # 黑色
            
            cv2.putText(canvas, window_text, (10, 30), font, font_scale, font_color, font_thickness)
            cv2.putText(canvas, positions_text, (10, 60), font, font_scale * 0.8, font_color, font_thickness - 1)
            if tokens_text:
                cv2.putText(canvas, tokens_text, (10, 85), font, font_scale * 0.7, font_color, font_thickness - 1)
            
            frame = canvas
        else:
            frame = overlay
        
        # 确保帧的尺寸匹配
        if frame.shape[0] != video_height or frame.shape[1] != video_width:
            frame = cv2.resize(frame, (video_width, video_height))
        
        # BGR格式（OpenCV要求）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 写入帧
        video_writer.write(frame_bgr)
    
    # 释放资源
    video_writer.release()
    
    print(f"[create_sliding_window_attention_video] Video saved to: {output_path}")
    print(f"[create_sliding_window_attention_video] Total frames: {len(sliding_window_results)}")
    print(f"[create_sliding_window_attention_video] Duration: {len(sliding_window_results) / fps:.2f} seconds")
    
    return str(output_path)
