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
            device_map=config.MODEL_DEVICE_MAP,
            attn_implementation="eager"  # Required for output_attentions support
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

        # Get actual model architecture info
        num_layers = len(state.model.model.language_model.layers)
        print(f"Model architecture: {num_layers} layers")
        print(f"Target layers for attention extraction: {state.extractor.target_layers}")

        print("Model loaded successfully!")
        return f"✓ Model loaded successfully! ({num_layers} layers)"

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
        Tuple of (generated_text, status_message, token_selector_update,
                 layer_start_update, layer_end_update, head_start_update, head_end_update,
                 token_range_display_html, token_range_start_idx, token_range_end_idx)
    """
    empty_html = "<p style='color: gray;'>No tokens to display</p>"
    
    if state.model is None:
        return ("", "Please load the model first!", gr.update(choices=[], value=None),
                gr.update(), gr.update(), gr.update(), gr.update(), 
                empty_html, 0, 0)

    if image is None:
        return ("", "Please upload an image!", gr.update(choices=[], value=None),
                gr.update(), gr.update(), gr.update(), gr.update(), 
                empty_html, 0, 0)

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

        # Get attention weights for ALL generation steps
        state.current_attention = state.extractor.get_all_generation_steps()
        
        # Debug: Print extracted attention info
        print(f"Extracted attention from {len(state.current_attention)} generation steps")
        if state.current_attention:
            step_keys = sorted(state.current_attention.keys())
            print(f"Step keys (sequence lengths): {step_keys}")
            
            sample_step = step_keys[0]
            sample_layer = list(state.current_attention[sample_step].keys())[0]
            num_layers = len(state.current_attention[sample_step])
            num_heads = len(state.current_attention[sample_step][sample_layer])
            
            print(f"Layer indices: {sorted(state.current_attention[sample_step].keys())}")
            print(f"Number of heads per layer: {num_heads}")
            
            # Show attention shape for each step
            print(f"\nAttention shapes by step:")
            for step in step_keys[:min(5, len(step_keys))]:  # Show first 5 steps
                sample_attn = state.current_attention[step][sample_layer][0]
                print(f"  Step {step}: {sample_attn.shape}")

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
            # Log attention info for new data structure
            print(f"\n{'='*60}")
            print(f"Attention Extraction Summary")
            print(f"{'='*60}")
            print(f"Number of generation steps: {len(state.current_attention)}")
            if state.current_attention:
                first_step = min(state.current_attention.keys())
                first_step_data = state.current_attention[first_step]
                print(f"Number of layers: {len(first_step_data)}")
                sample_layer = list(first_step_data.keys())[0]
                print(f"Number of heads: {len(first_step_data[sample_layer])}")
                sample_head = list(first_step_data[sample_layer].keys())[0]
                print(f"Sample attention shape: {first_step_data[sample_layer][sample_head].shape}")
            print(f"{'='*60}\n")
            
            info = state.current_processor.get_sequence_info()
            print(f"Sequence info: {info}")

        # Create token choices for dropdown
        token_choices = [
            f"Token {i}: '{tok}'"
            for i, tok in enumerate(state.current_tokens)
        ]

        status = f"✓ Generated {len(generated_ids)} tokens successfully!"

        # Return dropdown update with choices and select first token by default
        dropdown_update = gr.update(
            choices=token_choices,
            value=token_choices[0] if token_choices else None
        )
        
        # Get actual layer and head counts for UI updates
        # Use the first generation step to get layer/head info
        first_step = min(state.current_attention.keys())
        first_step_attn = state.current_attention[first_step]
        
        available_layers = sorted(first_step_attn.keys())
        sample_layer = available_layers[0]
        available_heads = sorted(first_step_attn[sample_layer].keys())
        
        max_layer = max(available_layers)
        max_head = max(available_heads)
        
        # Update layer sliders
        layer_start_update = gr.update(maximum=max_layer, value=max(0, max_layer - 4))
        layer_end_update = gr.update(maximum=max_layer, value=max_layer)
        
        # Update head sliders  
        head_start_update = gr.update(maximum=max_head, value=0)
        head_end_update = gr.update(maximum=max_head, value=max_head)
        
        # Set initial token range values
        num_tokens = len(state.current_tokens)
        initial_start = 0
        initial_end = min(4, num_tokens - 1)
        
        # Generate initial HTML display
        initial_html = generate_token_display_html(
            state.current_tokens,
            start_idx=initial_start,
            end_idx=initial_end
        )

        return (generated_text, status, dropdown_update, 
                layer_start_update, layer_end_update,
                head_start_update, head_end_update,
                initial_html, initial_start, initial_end)

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        error_html = "<p style='color: red;'>Error during generation</p>"
        return ("", f"✗ {error_msg}", gr.update(choices=[], value=None),
                gr.update(), gr.update(), gr.update(), gr.update(), 
                error_html, 0, 0)


# =============================================================================
# Visualization
# =============================================================================

def visualize_token_range_attention(
    token_start_idx: int,
    token_end_idx: int,
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
    Visualize aggregated attention for a range of tokens.
    
    Args:
        token_start_idx: Start token index (inclusive)
        token_end_idx: End token index (inclusive)
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
        print("Error: No attention data available. Please generate text first.")
        return None
    
    if token_start_idx < 0 or token_end_idx < 0:
        print("Error: Invalid token range")
        return None
    
    try:
        input_length = state.current_input_ids.shape[1]
        
        # Get available step keys
        available_steps = sorted(state.current_attention.keys())
        
        print(f"Visualizing token range: {token_start_idx} to {token_end_idx}")
        print(f"Input length: {input_length}")
        
        # Collect attention maps for all tokens in range
        attention_maps = []
        token_texts = []
        
        for token_idx in range(token_start_idx, token_end_idx + 1):
            if token_idx >= len(state.current_tokens):
                continue
                
            absolute_token_position = input_length + token_idx
            step_key = absolute_token_position
            
            # Find closest step if exact match not found
            if step_key not in state.current_attention:
                if not available_steps:
                    continue
                step_key = min(available_steps, key=lambda x: abs(x - step_key))
            
            step_attention = state.current_attention[step_key]
            
            # Get available layers and heads
            available_layers = sorted(step_attention.keys())
            if not available_layers:
                continue
            
            sample_layer = available_layers[0]
            available_heads = sorted(step_attention[sample_layer].keys())
            
            # Validate and constrain indices
            valid_layer_start = max(layer_start, min(available_layers))
            valid_layer_end = min(layer_end, max(available_layers))
            layer_indices = [l for l in range(valid_layer_start, valid_layer_end + 1) if l in available_layers]
            
            valid_head_start = max(head_start, min(available_heads))
            valid_head_end = min(head_end, max(available_heads))
            head_indices = [h for h in range(valid_head_start, valid_head_end + 1) if h in available_heads]
            
            # Get attention heatmap for this token
            attention_map = state.current_processor.get_attention_heatmap_for_token(
                step_attention,
                token_position=absolute_token_position,
                layer_indices=layer_indices,
                head_indices=head_indices,
                aggregation_method=aggregation_method.lower(),
                normalize=True
            )
            
            if attention_map is not None:
                attention_maps.append(attention_map)
                token_texts.append(state.current_tokens[token_idx])
        
        if not attention_maps:
            print("ERROR: No attention maps generated for the token range")
            return None
        
        print(f"Successfully generated {len(attention_maps)} attention maps")
        
        # Aggregate attention maps
        aggregated_map = np.mean(attention_maps, axis=0)
        
        # Create visualization
        visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)
        
        token_range_text = f"Tokens {token_start_idx}-{token_end_idx}: '{' '.join(token_texts)}'"
        
        if show_comparison:
            result = visualizer.create_side_by_side_comparison(
                state.current_image,
                aggregated_map,
                titles=("Original Image", f"Attention: {token_range_text}")
            )
        else:
            result = visualizer.create_heatmap_overlay(
                state.current_image,
                aggregated_map
            )
        
        print("Visualization created successfully!")
        return result
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


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
    # Check if generation has been done
    if state.current_attention is None or state.current_processor is None:
        print("Error: No attention data available. Please generate text first.")
        return None

    # Check if a token is selected
    if token_selector is None or token_selector == "":
        print("Error: No token selected. Please select a token from the dropdown.")
        return None

    try:
        # Extract token index from selector string
        # Format: "Token {i}: '{tok}'"
        print(f"Visualizing token: {token_selector}")
        token_idx = int(token_selector.split(":")[0].split()[-1])

        # Adjust token position (account for input tokens)
        input_length = state.current_input_ids.shape[1]
        absolute_token_position = input_length + token_idx
        
        # The step key is the total sequence length at the time this token was generated
        # = input_length + token_idx + 1 (because we're generating the (token_idx+1)-th new token)
        step_key = absolute_token_position
        
        print(f"Token index: {token_idx}, Input length: {input_length}, Absolute position: {absolute_token_position}")
        print(f"Looking for attention at step key (sequence length): {step_key}")
        print(f"Available step keys: {sorted(state.current_attention.keys())}")
        
        # Get attention for this specific generation step
        if step_key not in state.current_attention:
            print(f"ERROR: Step key {step_key} not found in attention data")
            # Try to find the closest step
            available_keys = sorted(state.current_attention.keys())
            if available_keys:
                closest_key = min(available_keys, key=lambda x: abs(x - step_key))
                print(f"Using closest available step: {closest_key}")
                step_attention = state.current_attention[closest_key]
            else:
                print("ERROR: No attention data available")
                return None
        else:
            step_attention = state.current_attention[step_key]

        # Get available layers and heads from this step
        available_layers = sorted(step_attention.keys())
        if not available_layers:
            print("ERROR: No attention data available")
            return None
        
        sample_layer = available_layers[0]
        available_heads = sorted(step_attention[sample_layer].keys())
        
        print(f"Available layers: {available_layers}")
        print(f"Available heads: {available_heads}")
        print(f"Requested layer range: {layer_start}-{layer_end}")
        print(f"Requested head range: {head_start}-{head_end}")
        
        # Validate and constrain layer indices
        valid_layer_start = max(layer_start, min(available_layers))
        valid_layer_end = min(layer_end, max(available_layers))
        
        if valid_layer_start > valid_layer_end:
            print(f"ERROR: Invalid layer range. Available: {min(available_layers)}-{max(available_layers)}")
            return None
            
        layer_indices = list(range(valid_layer_start, valid_layer_end + 1))
        layer_indices = [l for l in layer_indices if l in available_layers]
        
        # Validate and constrain head indices
        valid_head_start = max(head_start, min(available_heads))
        valid_head_end = min(head_end, max(available_heads))
        
        if valid_head_start > valid_head_end:
            print(f"ERROR: Invalid head range. Available: {min(available_heads)}-{max(available_heads)}")
            return None
            
        head_indices = list(range(valid_head_start, valid_head_end + 1))
        head_indices = [h for h in head_indices if h in available_heads]
        
        print(f"Using layers: {layer_indices}")
        print(f"Using heads: {head_indices}")

        # Debug processor info
        print(f"Processor info:")
        print(f"  - Has patch_mapping: {state.current_processor.patch_mapping is not None}")
        print(f"  - Has image_grid_thw: {state.current_processor.image_grid_thw is not None}")
        print(f"  - Vision token ranges: {state.current_processor.vision_token_ranges}")
        
        if state.current_processor.image_grid_thw is not None:
            print(f"  - Image grid THW: {state.current_processor.image_grid_thw}")
        
        # Get attention heatmap using the attention from this specific generation step
        print("Getting attention heatmap...")
        attention_map = state.current_processor.get_attention_heatmap_for_token(
            step_attention,
            token_position=absolute_token_position,
            layer_indices=layer_indices,
            head_indices=head_indices,
            aggregation_method=aggregation_method.lower(),
            normalize=True
        )

        if attention_map is None:
            print("ERROR: No attention map generated")
            print("Possible causes:")
            print("  1. No vision tokens found in sequence")
            print("  2. Patch mapping failed")
            print("  3. Token position out of range")
            return None

        print(f"Attention map shape: {attention_map.shape}")

        # Create visualization
        print(f"Creating visualization with colormap={colormap}, alpha={alpha}")
        visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)

        if show_comparison:
            print("Creating side-by-side comparison...")
            result = visualizer.create_side_by_side_comparison(
                state.current_image,
                attention_map,
                titles=("Original Image", f"Attention: {token_selector}")
            )
        else:
            print("Creating heatmap overlay...")
            result = visualizer.create_heatmap_overlay(
                state.current_image,
                attention_map
            )

        print("Visualization created successfully!")
        return result

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Token Selection HTML Generator
# =============================================================================

def generate_token_display_html(tokens, start_idx=None, end_idx=None):
    """
    Generate static HTML displaying tokens with highlighting.
    
    Args:
        tokens: List of token strings
        start_idx: Start index of selection (None if not selected)
        end_idx: End index of selection (None if not selected)
        
    Returns:
        HTML string with token display
    """
    if not tokens:
        return "<p style='color: gray;'>No tokens to display</p>"
    
    html = """
    <style>
        .token-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
            border: 2px solid #dee2e6;
        }
        .token-box {
            padding: 8px 14px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            background-color: white;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            display: inline-block;
        }
        .token-box.start {
            background-color: #0066cc;
            color: white;
            border-color: #0052a3;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,102,204,0.3);
        }
        .token-box.end {
            background-color: #0066cc;
            color: white;
            border-color: #0052a3;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,102,204,0.3);
        }
        .token-box.in-range {
            background-color: #cce5ff;
            color: #003d7a;
            border-color: #99ccff;
            font-weight: 500;
        }
        .token-index {
            font-size: 10px;
            color: #6c757d;
            margin-right: 6px;
            font-weight: normal;
        }
        .token-box.start .token-index,
        .token-box.end .token-index {
            color: #ffffff;
        }
        .token-box.in-range .token-index {
            color: #0052a3;
        }
    </style>
    
    <div class="token-container">
    """
    
    for i, token in enumerate(tokens):
        # Escape HTML special characters
        token_escaped = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
        
        # Determine class based on selection
        token_class = "token-box"
        if start_idx is not None and end_idx is not None:
            if i == start_idx:
                token_class += " start"
            elif i == end_idx:
                token_class += " end"
            elif start_idx < i < end_idx:
                token_class += " in-range"
        
        html += f"""
        <div class="{token_class}">
            <span class="token-index">[{i}]</span>
            <span class="token-text">{token_escaped}</span>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html


# =============================================================================
# Unified Visualization Interface
# =============================================================================

def update_token_range_display_manual(start_idx: int, end_idx: int) -> str:
    """
    Update the HTML display when manual input changes.
    
    Args:
        start_idx: Start token index
        end_idx: End token index
        
    Returns:
        Updated HTML string
    """
    if state.current_tokens is None or len(state.current_tokens) == 0:
        return "<p style='color: gray;'>No tokens available</p>"
    
    try:
        # Validate indices
        start_idx = max(0, min(int(start_idx), len(state.current_tokens) - 1))
        end_idx = max(0, min(int(end_idx), len(state.current_tokens) - 1))
        
        # Ensure start <= end
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        return generate_token_display_html(state.current_tokens, start_idx, end_idx)
    except Exception as e:
        print(f"Error updating token display: {e}")
        return generate_token_display_html(state.current_tokens, 0, 0)


def visualize_attention_unified(
    selection_mode: str,
    token_selector: Optional[str],
    token_range_start_idx: int,
    token_range_end_idx: int,
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
    Unified visualization function that handles both single token and token range.
    
    Args:
        selection_mode: "Single Token" or "Token Range"
        token_selector: Selected token string (for single token mode)
        token_range_start_idx: Start token index (for range mode)
        token_range_end_idx: End token index (for range mode)
        ... other parameters
        
    Returns:
        PIL Image with visualization
    """
    if selection_mode == "Single Token":
        # Use single token visualization
        return visualize_token_attention(
            token_selector=token_selector,
            layer_start=layer_start,
            layer_end=layer_end,
            head_start=head_start,
            head_end=head_end,
            aggregation_method=aggregation_method,
            colormap=colormap,
            alpha=alpha,
            show_comparison=show_comparison
        )
    else:
        # Use token range visualization with provided indices
        try:
            token_start_idx = int(token_range_start_idx)
            token_end_idx = int(token_range_end_idx)
            
            if token_start_idx > token_end_idx:
                print("WARNING: Start token is after end token, swapping them")
                token_start_idx, token_end_idx = token_end_idx, token_start_idx
            
            # Use token range visualization
            return visualize_token_range_attention(
                token_start_idx=token_start_idx,
                token_end_idx=token_end_idx,
                layer_start=layer_start,
                layer_end=layer_end,
                head_start=head_start,
                head_end=head_end,
                aggregation_method=aggregation_method,
                colormap=colormap,
                alpha=alpha,
                show_comparison=show_comparison
            )
        except Exception as e:
            print(f"ERROR: Failed to visualize token range: {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface():
    """Create the Gradio interface."""

    # Get model info - will be updated after model loads
    # These are placeholder values, actual values set dynamically
    num_layers, num_heads = 28, 28  # Default for 7B model (will be updated)

    with gr.Blocks(title="Qwen2.5-VL Attention Visualizer") as demo:
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
                # Selection mode
                selection_mode = gr.Radio(
                    ["Single Token", "Token Range"],
                    value="Single Token",
                    label="Selection Mode",
                    info="Choose to visualize a single token or aggregate attention over multiple tokens"
                )
                
                # Single token selector
                token_selector = gr.Dropdown(
                    label="Select Token",
                    choices=[],
                    interactive=True,
                    visible=True
                )
                
                # Token range selector
                with gr.Column(visible=False) as token_range_col:
                    # Token visualization display
                    token_range_display = gr.HTML(
                        value="<p style='color: gray;'>Generate text first to see tokens</p>"
                    )
                    
                    # Range input fields
                    with gr.Row():
                        token_range_start_input = gr.Number(
                            label="Start Index",
                            value=0,
                            precision=0,
                            minimum=0,
                            interactive=True
                        )
                        token_range_end_input = gr.Number(
                            label="End Index",
                            value=0,
                            precision=0,
                            minimum=0,
                            interactive=True
                        )
                        update_display_btn = gr.Button(
                            "Update", 
                            size="sm"
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

        1. **Load Model**: Click "Load Model"
        2. **Generate**: Upload image and click "Generate"  
        3. **Visualize**: 
           - **Single Token**: Select a token from dropdown
           - **Token Range**: Enter start/end indices, click Update, then Visualize
        """)

        # Event handlers
        load_btn.click(
            fn=load_model,
            outputs=load_status
        )

        generate_btn.click(
            fn=generate_with_attention,
            inputs=[image_input, prompt_input, max_tokens, temperature, top_p],
            outputs=[output_text, gen_status, token_selector, 
                    layer_start, layer_end, head_start, head_end,
                    token_range_display, token_range_start_input, token_range_end_input]
        )
        
        # Toggle visibility based on selection mode
        def toggle_selection_mode(mode):
            if mode == "Single Token":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        selection_mode.change(
            fn=toggle_selection_mode,
            inputs=[selection_mode],
            outputs=[token_selector, token_range_col]
        )
        
        # Update HTML display when update button is clicked
        update_display_btn.click(
            fn=update_token_range_display_manual,
            inputs=[token_range_start_input, token_range_end_input],
            outputs=[token_range_display]
        )

        visualize_btn.click(
            fn=visualize_attention_unified,
            inputs=[
                selection_mode, token_selector, 
                token_range_start_input, token_range_end_input,
                layer_start, layer_end, head_start, head_end, 
                aggregation, colormap, alpha_slider, show_comparison
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
