"""
Gradio Interface for Qwen2.5-VL Attention Visualization

This is the main application that provides a web interface for
visualizing attention patterns in Qwen2.5-VL during text generation.
"""

import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple, Dict
import gc

import config
import utils
from attention_extractor import AttentionExtractor
from attention_processor import AttentionProcessor
from visualization import AttentionVisualizer

# =============================================================================
# Global Model State
# =============================================================================

class ModelState:
    """Global state to hold loaded model and current generation results."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.extractor = None
        self.current_attention = None
        self.current_processor = None
        self.current_tokens = None
        self.current_image = None
        self.current_input_ids = None
        self.current_image_grid_thw = None
        self.generated_text = None

    def reset(self):
        """Reset generation-specific state."""
        self.current_attention = None
        self.current_processor = None
        self.current_tokens = None
        self.current_input_ids = None
        self.current_image_grid_thw = None
        self.generated_text = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global state instance
state = ModelState()


# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    """Load Qwen2.5-VL model and processor."""
    if state.model is not None:
        return "Model already loaded!"

    try:
        print(f"Loading model: {config.MODEL_NAME}")
        state.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=config.MODEL_TORCH_DTYPE,
            device_map=config.MODEL_DEVICE_MAP
        )

        print(f"Loading processor: {config.PROCESSOR_NAME}")
        state.processor = AutoProcessor.from_pretrained(config.PROCESSOR_NAME)
        state.tokenizer = state.processor.tokenizer

        # Create attention extractor
        state.extractor = AttentionExtractor(
            state.model,
            extract_all_layers=config.EXTRACT_ALL_LAYERS,
            specific_layers=config.SPECIFIC_LAYERS
        )

        print("Model loaded successfully!")
        return "✓ Model loaded successfully!"

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        return f"✗ {error_msg}"


# =============================================================================
# Generation with Attention Extraction
# =============================================================================

def generate_with_attention(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
    temperature: float = config.DEFAULT_TEMPERATURE,
    top_p: float = config.DEFAULT_TOP_P
) -> Tuple[str, str, Dict]:
    """
    Generate text from image with attention extraction.

    Returns:
        Tuple of (generated_text, status_message, token_info_dict)
    """
    if state.model is None:
        return "", "Please load the model first!", {}

    if image is None:
        return "", "Please upload an image!", {}

    try:
        # Reset previous state
        state.reset()
        state.current_image = image

        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = state.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process with qwen_vl_utils
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = state.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        device = state.model.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Store input info
        state.current_input_ids = inputs["input_ids"]
        state.current_image_grid_thw = inputs.get("image_grid_thw")

        # Generate with attention extraction
        print("Generating with attention extraction...")
        state.extractor.start_extraction()

        with torch.no_grad():
            output_ids = state.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )

        state.extractor.stop_extraction()

        # Get attention weights
        state.current_attention = state.extractor.get_attention_weights()

        # Decode output
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = state.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        state.generated_text = generated_text

        # Get individual tokens
        state.current_tokens = utils.decode_tokens_to_text(
            generated_ids.tolist(),
            state.tokenizer
        )

        # Create attention processor
        state.current_processor = AttentionProcessor(
            input_ids=inputs["input_ids"],
            image_grid_thw=state.current_image_grid_thw,
            original_image_size=(image.width, image.height),
            resized_image_size=None  # Will be inferred
        )

        # Log info
        if config.VERBOSE:
            utils.log_attention_info(state.current_attention)
            info = state.current_processor.get_sequence_info()
            print(f"Sequence info: {info}")

        # Create token choices for dropdown
        token_choices = {
            f"Token {i}: '{tok}'": i
            for i, tok in enumerate(state.current_tokens)
        }

        status = f"✓ Generated {len(generated_ids)} tokens successfully!"

        return generated_text, status, token_choices

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return "", f"✗ {error_msg}", {}


# =============================================================================
# Visualization
# =============================================================================

def visualize_token_attention(
    token_selector: str,
    layer_start: int,
    layer_end: int,
    head_start: int,
    head_end: int,
    aggregation_method: str,
    colormap: str,
    alpha: float,
    show_comparison: bool
) -> Optional[Image.Image]:
    """
    Visualize attention for a selected token.

    Args:
        token_selector: Selected token from dropdown
        layer_start, layer_end: Layer range
        head_start, head_end: Head range
        aggregation_method: How to aggregate attention
        colormap: Colormap name
        alpha: Overlay transparency
        show_comparison: Whether to show side-by-side comparison

    Returns:
        PIL Image with visualization
    """
    if state.current_attention is None or state.current_processor is None:
        return None

    try:
        # Extract token index from selector string
        # Format: "Token {i}: '{tok}'"
        token_idx = int(token_selector.split(":")[0].split()[-1])

        # Adjust token position (account for input tokens)
        input_length = state.current_input_ids.shape[1]
        absolute_token_position = input_length + token_idx

        # Get layer and head indices
        layer_indices = list(range(layer_start, layer_end + 1)) if layer_end >= layer_start else None
        head_indices = list(range(head_start, head_end + 1)) if head_end >= head_start else None

        # Get attention heatmap
        attention_map = state.current_processor.get_attention_heatmap_for_token(
            state.current_attention,
            token_position=absolute_token_position,
            layer_indices=layer_indices,
            head_indices=head_indices,
            aggregation_method=aggregation_method.lower(),
            normalize=True
        )

        if attention_map is None:
            print("No attention map generated")
            return None

        # Create visualization
        visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)

        if show_comparison:
            result = visualizer.create_side_by_side_comparison(
                state.current_image,
                attention_map,
                titles=("Original Image", f"Attention: {token_selector}")
            )
        else:
            result = visualizer.create_heatmap_overlay(
                state.current_image,
                attention_map
            )

        return result

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface():
    """Create the Gradio interface."""

    # Get model info
    num_layers, num_heads = 80, 64  # Default for 3B model

    with gr.Blocks(title="Qwen2.5-VL Attention Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Qwen2.5-VL Attention Visualization Tool

        Upload an image, enter a prompt, and explore how the model attends to different image regions
        while generating each token.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Step 1: Load Model")
                load_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Step 2: Input")
                image_input = gr.Image(type="pil", label="Upload Image")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    value="Describe this image in detail.",
                    lines=3
                )

                with gr.Accordion("Generation Settings", open=False):
                    max_tokens = gr.Slider(16, 512, value=128, step=16, label="Max New Tokens")
                    temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                gen_status = gr.Textbox(label="Generation Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 💬 Generated Text")
                output_text = gr.Textbox(
                    label="Output",
                    interactive=False,
                    lines=8
                )

        gr.Markdown("---")
        gr.Markdown("### 🎨 Step 3: Visualize Attention")

        with gr.Row():
            with gr.Column(scale=1):
                token_selector = gr.Dropdown(
                    label="Select Token",
                    choices=[],
                    interactive=True
                )

                with gr.Accordion("Attention Settings", open=True):
                    with gr.Row():
                        layer_start = gr.Slider(
                            0, num_layers-1, value=max(0, num_layers-5),
                            step=1, label="Layer Start"
                        )
                        layer_end = gr.Slider(
                            0, num_layers-1, value=num_layers-1,
                            step=1, label="Layer End"
                        )

                    with gr.Row():
                        head_start = gr.Slider(
                            0, num_heads-1, value=0,
                            step=1, label="Head Start"
                        )
                        head_end = gr.Slider(
                            0, num_heads-1, value=num_heads-1,
                            step=1, label="Head End"
                        )

                    aggregation = gr.Radio(
                        ["Mean", "Max", "Min"],
                        value="Mean",
                        label="Aggregation Method"
                    )

                with gr.Accordion("Visualization Settings", open=True):
                    colormap = gr.Dropdown(
                        ["jet", "hot", "viridis", "plasma", "coolwarm", "magma"],
                        value="jet",
                        label="Colormap"
                    )
                    alpha_slider = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="Overlay Transparency"
                    )
                    show_comparison = gr.Checkbox(
                        label="Show Side-by-Side Comparison",
                        value=False
                    )

                visualize_btn = gr.Button("Visualize", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_viz = gr.Image(label="Attention Visualization")

        gr.Markdown("""
        ---
        ### 📖 How to Use

        1. **Load Model**: Click "Load Model" to initialize Qwen2.5-VL
        2. **Generate**: Upload an image and prompt, then click "Generate"
        3. **Explore**: Select any generated token and click "Visualize" to see attention heatmap
        4. **Customize**: Adjust layer/head ranges and visualization settings to explore different patterns

        ### 💡 Tips

        - **Layers**: Later layers (60-80) often show more semantic attention
        - **Heads**: Different heads may focus on different aspects
        - **Aggregation**: "Mean" gives overall pattern, "Max" highlights peaks
        """)

        # Event handlers
        load_btn.click(
            fn=load_model,
            outputs=load_status
        )

        generate_btn.click(
            fn=generate_with_attention,
            inputs=[image_input, prompt_input, max_tokens, temperature, top_p],
            outputs=[output_text, gen_status, token_selector]
        )

        visualize_btn.click(
            fn=visualize_token_attention,
            inputs=[
                token_selector, layer_start, layer_end,
                head_start, head_end, aggregation,
                colormap, alpha_slider, show_comparison
            ],
            outputs=output_viz
        )

    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Qwen2.5-VL Attention Visualization Tool")
    print("="*60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.MODEL_DEVICE_MAP}")
    print(f"Server: {config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}")
    print("="*60)

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE
    )
