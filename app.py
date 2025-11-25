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

def load_model(model_path: str):
    """Load Qwen2.5-VL model and processor."""
    if state.model is not None:
        return "Model already loaded! Please restart the app to load a different model."

    if not model_path or model_path.strip() == "":
        return "✗ Please enter a valid model path!"

    try:
        print(f"Loading model: {model_path}")
        state.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=config.MODEL_TORCH_DTYPE,
            device_map=config.MODEL_DEVICE_MAP,
            attn_implementation="eager"  # Required for output_attentions support
        )

        print(f"Loading processor: {model_path}")
        state.processor = AutoProcessor.from_pretrained(model_path)
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
        return f"✓ Model loaded successfully from {model_path}! ({num_layers} layers)"

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
        Tuple of (generated_text, status_message,
                 token_range_display_html, token_range_start_idx, token_range_end_idx)
    """
    empty_html = "<p style='color: gray;'>No tokens to display</p>"
    
    if state.model is None:
        return ("", "Please load the model first!",
                empty_html, gr.update(), gr.update(), gr.update())

    if image is None:
        return ("", "Please upload an image!",
                empty_html, gr.update(), gr.update(), gr.update())

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

        status = f"✓ Generated {len(generated_ids)} tokens successfully!"
        
        # Set initial token range values
        num_tokens = len(state.current_tokens)
        initial_start = 0
        initial_end = min(4, num_tokens - 1)
        
        # Generate initial HTML display
        display_html = generate_token_range_display_html(
            state.current_tokens,
            start_idx=initial_start,
            end_idx=initial_end,
            click_count=2
        )
        
        # Update slider configs
        slider_max = num_tokens - 1
        token_index_update = gr.update(maximum=slider_max, value=0)
        start_slider_update = gr.update(maximum=slider_max, value=initial_start)
        end_slider_update = gr.update(maximum=slider_max, value=initial_end)

        return (generated_text, status, 
                display_html,
                token_index_update, start_slider_update, end_slider_update)

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        error_html = "<p style='color: red;'>Error during generation</p>"
        return ("", f"✗ {error_msg}",
                error_html, gr.update(), gr.update(), gr.update())


# =============================================================================
# Visualization
# =============================================================================

def visualize_token_range_attention(
    token_start_idx: int,
    token_end_idx: int,
    aggregation_method: str,
    colormap: str,
    alpha: float
) -> Optional[Dict]:
    """
    Visualize attention for a range of tokens across all layers.
    Returns attention maps for all 28 layers + 1 mean map.
    
    Args:
        token_start_idx: Start token index (inclusive)
        token_end_idx: End token index (inclusive)
        aggregation_method: How to aggregate attention heads (mean/max/min)
        colormap: Colormap name
        alpha: Overlay transparency
        
    Returns:
        Dictionary mapping layer names to attention heatmaps
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
        
        # Collect attention maps per layer for all tokens in range
        # layer_attention_maps[layer_idx] = list of attention maps for this layer across tokens
        layer_attention_maps = {}
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
            
            # For each layer, get attention map with head aggregation
            for layer_idx in available_layers:
                # Get attention heatmap for this token and this layer
                attention_map = state.current_processor.get_attention_heatmap_for_token(
                    step_attention,
                    token_position=absolute_token_position,
                    layer_indices=[layer_idx],  # Single layer
                    head_indices=available_heads,  # All heads
                    aggregation_method=aggregation_method.lower(),
                    normalize=True
                )
                
                if attention_map is not None:
                    if layer_idx not in layer_attention_maps:
                        layer_attention_maps[layer_idx] = []
                    layer_attention_maps[layer_idx].append(attention_map)
            
            if token_idx == token_start_idx or len(token_texts) < 10:  # Store first few token texts
                token_texts.append(state.current_tokens[token_idx])
        
        if not layer_attention_maps:
            print("ERROR: No attention maps generated for the token range")
            return None
        
        print(f"Successfully generated attention maps for {len(layer_attention_maps)} layers")
        
        # Average attention maps across tokens for each layer
        layer_averaged_maps = {}
        for layer_idx, maps in layer_attention_maps.items():
            layer_averaged_maps[layer_idx] = np.mean(maps, axis=0)
        
        # Calculate mean across all layers
        all_layers_mean = np.mean(list(layer_averaged_maps.values()), axis=0)
        
        # Return a dictionary with layer indices as keys
        sorted_layers = sorted(layer_averaged_maps.keys())
        result_dict = {}
        for layer_idx in sorted_layers:
            result_dict[f"Layer {layer_idx}"] = layer_averaged_maps[layer_idx]
        result_dict["Mean (All Layers)"] = all_layers_mean
        
        print(f"Visualization created successfully! Generated attention maps for {len(result_dict)} layers")
        return result_dict
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def visualize_token_attention(
    token_selector: str,
    aggregation_method: str,
    colormap: str,
    alpha: float
) -> Optional[Dict]:
    """
    Visualize attention for a selected token across all layers.
    Returns attention maps for all 28 layers + 1 mean map.

    Args:
        token_selector: Selected token from dropdown
        aggregation_method: How to aggregate attention heads (mean/max/min)
        colormap: Colormap name
        alpha: Overlay transparency

    Returns:
        Dictionary mapping layer names to attention heatmaps
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

        # Debug processor info
        print(f"Processor info:")
        print(f"  - Has patch_mapping: {state.current_processor.patch_mapping is not None}")
        print(f"  - Has image_grid_thw: {state.current_processor.image_grid_thw is not None}")
        print(f"  - Vision token ranges: {state.current_processor.vision_token_ranges}")
        
        if state.current_processor.image_grid_thw is not None:
            print(f"  - Image grid THW: {state.current_processor.image_grid_thw}")
        
        # Get attention heatmap for each layer
        layer_attention_maps = {}
        for layer_idx in available_layers:
            print(f"Getting attention heatmap for layer {layer_idx}...")
            attention_map = state.current_processor.get_attention_heatmap_for_token(
                step_attention,
                token_position=absolute_token_position,
                layer_indices=[layer_idx],  # Single layer
                head_indices=available_heads,  # All heads
                aggregation_method=aggregation_method.lower(),
                normalize=True
            )
            
            if attention_map is not None:
                layer_attention_maps[layer_idx] = attention_map

        if not layer_attention_maps:
            print("ERROR: No attention map generated")
            print("Possible causes:")
            print("  1. No vision tokens found in sequence")
            print("  2. Patch mapping failed")
            print("  3. Token position out of range")
            return None

        print(f"Generated attention maps for {len(layer_attention_maps)} layers")
        
        # Calculate mean across all layers
        all_layers_mean = np.mean(list(layer_attention_maps.values()), axis=0)

        # Return a dictionary with layer indices as keys
        sorted_layers = sorted(layer_attention_maps.keys())
        result_dict = {}
        for layer_idx in sorted_layers:
            result_dict[f"Layer {layer_idx}"] = layer_attention_maps[layer_idx]
        result_dict["Mean (All Layers)"] = all_layers_mean

        print(f"Visualization created successfully! Generated attention maps for {len(result_dict)} layers")
        return result_dict

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Token Selection HTML Generator
# =============================================================================

def generate_single_token_display_html(tokens, selected_idx=None):
    """
    Generate interactive HTML for Single Token mode.
    
    Args:
        tokens: List of token strings
        selected_idx: Currently selected token index
        
    Returns:
        HTML string with token display
    """
    if not tokens:
        return "<p style='color: gray;'>No tokens to display</p>"
    
    # Generate token selector buttons as HTML
    html_parts = []
    html_parts.append("""
    <style>
        .token-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        .token-btn {
            padding: 8px 14px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            background-color: white;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .token-btn:hover {
            background-color: #e7f3ff;
            border-color: #0066cc;
            transform: translateY(-2px);
        }
        .token-btn.selected {
            background-color: #28a745;
            color: white;
            border-color: #218838;
            font-weight: bold;
        }
        .token-idx {
            font-size: 10px;
            color: #6c757d;
            margin-right: 6px;
        }
        .token-btn.selected .token-idx {
            color: #ffffff;
        }
    </style>
    """)
    
    if selected_idx is None:
        html_parts.append('<p style="margin-bottom: 10px; color: #155724;">💡 Select a token from dropdown or use manual input</p>')
    else:
        html_parts.append(f'<p style="margin-bottom: 10px; color: #155724;">✓ Token [{selected_idx}] selected</p>')
    
    html_parts.append('<div class="token-grid">')
    for i, token in enumerate(tokens):
        token_escaped = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        css_class = "token-btn selected" if i == selected_idx else "token-btn"
        html_parts.append(f'<div class="{css_class}"><span class="token-idx">[{i}]</span>{token_escaped}</div>')
    html_parts.append('</div>')
    
    return ''.join(html_parts)


def generate_token_range_display_html(tokens, start_idx=None, end_idx=None, click_count=0):
    """
    Generate interactive HTML for Token Range mode.
    
    Args:
        tokens: List of token strings
        start_idx: Start index of selection (None if not selected)
        end_idx: End index of selection (None if not selected)
        click_count: Number of clicks made (0, 1, or 2+)
        
    Returns:
        HTML string with token display
    """
    if not tokens:
        return "<p style='color: gray;'>No tokens to display</p>"
    
    html_parts = []
    html_parts.append("""
    <style>
        .token-grid-range {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        .token-btn-range {
            padding: 8px 14px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            background-color: white;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            transition: all 0.2s ease;
        }
        .token-btn-range.start,
        .token-btn-range.end {
            background-color: #0066cc;
            color: white;
            border-color: #0052a3;
            font-weight: bold;
        }
        .token-btn-range.in-range {
            background-color: #cce5ff;
            color: #003d7a;
            border-color: #99ccff;
        }
        .token-idx-range {
            font-size: 10px;
            color: #6c757d;
            margin-right: 6px;
        }
        .token-btn-range.start .token-idx-range,
        .token-btn-range.end .token-idx-range {
            color: #ffffff;
        }
        .token-btn-range.in-range .token-idx-range {
            color: #0052a3;
        }
    </style>
    """)
    
    # Add hint message
    if click_count == 0:
        html_parts.append('<p style="margin-bottom: 10px; padding: 10px; background-color: #e7f3ff; border-left: 4px solid #0066cc; border-radius: 4px;">💡 Use manual input below to set <strong>Start Index</strong></p>')
    elif click_count == 1:
        html_parts.append('<p style="margin-bottom: 10px; padding: 10px; background-color: #e7f3ff; border-left: 4px solid #0066cc; border-radius: 4px;">💡 Now set <strong>End Index</strong></p>')
    else:
        html_parts.append('<p style="margin-bottom: 10px; padding: 10px; background-color: #e7f3ff; border-left: 4px solid #0066cc; border-radius: 4px;">✓ Range selected. You can adjust values below.</p>')
    
    html_parts.append('<div class="token-grid-range">')
    
    for i, token in enumerate(tokens):
        token_escaped = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        css_class = "token-btn-range"
        if start_idx is not None and end_idx is not None:
            if i == start_idx:
                css_class += " start"
            elif i == end_idx:
                css_class += " end"
            elif min(start_idx, end_idx) < i < max(start_idx, end_idx):
                css_class += " in-range"
        elif start_idx is not None and click_count == 1:
            if i == start_idx:
                css_class += " start"
        
        html_parts.append(f'<div class="{css_class}"><span class="token-idx-range">[{i}]</span>{token_escaped}</div>')
    
    html_parts.append('</div>')
    
    return ''.join(html_parts)


# =============================================================================
# Layer Visualization Helper
# =============================================================================

# Global storage for attention maps
current_attention_maps = None

def create_layer_visualization(
    layer_name: str,
    colormap: str,
    alpha: float
) -> Optional[Image.Image]:
    """
    Create visualization for a specific layer.
    
    Args:
        layer_name: Name of layer to visualize (e.g., "Layer 0" or "Mean (All Layers)")
        colormap: Colormap name
        alpha: Overlay transparency
        
    Returns:
        PIL Image with visualization
    """
    global current_attention_maps
    
    if current_attention_maps is None:
        return None
    
    if layer_name not in current_attention_maps:
        return None
    
    if state.current_image is None:
        return None
    
    try:
        visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)
        attention_map = current_attention_maps[layer_name]
        
        result = visualizer.create_heatmap_overlay(
            state.current_image,
            attention_map
        )
        
        return result
    except Exception as e:
        print(f"Error creating layer visualization: {e}")
        return None


# =============================================================================
# Unified Visualization Interface
# =============================================================================

def handle_single_token_click(token_idx):
    """Handle clicking a token in Single Token mode."""
    if state.current_tokens is None or len(state.current_tokens) == 0:
        return gr.update(), "<p style='color: gray;'>No tokens available</p>"
    
    # Update dropdown selection
    token_choices = [f"Token {i}: '{tok}'" for i, tok in enumerate(state.current_tokens)]
    if 0 <= token_idx < len(token_choices):
        dropdown_value = token_choices[token_idx]
        html = generate_single_token_display_html(state.current_tokens, token_idx)
        return gr.update(value=dropdown_value), html
    return gr.update(), generate_single_token_display_html(state.current_tokens, None)


def handle_range_token_click(token_idx, current_start, current_end, click_count):
    """Handle clicking a token in Token Range mode."""
    if state.current_tokens is None or len(state.current_tokens) == 0:
        return None, None, 0, "<p style='color: gray;'>No tokens available</p>"
    
    try:
        # Validate token index
        if token_idx < 0 or token_idx >= len(state.current_tokens):
            return current_start, current_end, click_count, generate_token_range_display_html(
                state.current_tokens, current_start, current_end, click_count
            )
        
        # Determine what to do based on click count
        if click_count == 0 or click_count >= 2:
            # First click or reset - set start
            new_start = token_idx
            new_end = None
            new_click_count = 1
        elif click_count == 1:
            # Second click - set end
            new_start = current_start
            new_end = token_idx
            new_click_count = 2
        else:
            new_start = current_start
            new_end = current_end
            new_click_count = click_count
        
        # Generate updated HTML
        html = generate_token_range_display_html(
            state.current_tokens, 
            new_start, 
            new_end, 
            new_click_count
        )
        
        return new_start, new_end, new_click_count, html
        
    except Exception as e:
        print(f"Error handling range token click: {e}")
        import traceback
        traceback.print_exc()
        return current_start, current_end, click_count, generate_token_range_display_html(
            state.current_tokens, current_start, current_end, click_count
        )


def update_token_range_display_manual(start_idx, end_idx, click_state):
    """
    Update the HTML display when manual input changes.
    
    Args:
        start_idx: Start token index (can be None)
        end_idx: End token index (can be None)
        click_state: Current click count state
        
    Returns:
        Updated HTML string and updated click state
    """
    if state.current_tokens is None or len(state.current_tokens) == 0:
        return "<p style='color: gray;'>No tokens available</p>", 0
    
    try:
        # Handle None values
        if start_idx is None or start_idx == '':
            start_idx = None
            click_count = 0
        else:
            start_idx = max(0, min(int(start_idx), len(state.current_tokens) - 1))
            click_count = 1 if (end_idx is None or end_idx == '') else 2
        
        if end_idx is None or end_idx == '':
            end_idx = None
        else:
            end_idx = max(0, min(int(end_idx), len(state.current_tokens) - 1))
            click_count = 2
        
        # If both are set, use click_state to determine if we should show full selection
        if start_idx is not None and end_idx is not None:
            click_count = click_state if click_state > 0 else 2
        
        html = generate_token_range_display_html(
            state.current_tokens, 
            start_idx, 
            end_idx, 
            click_count
        )
        
        return html, click_count
        
    except Exception as e:
        print(f"Error updating token display: {e}")
        import traceback
        traceback.print_exc()
        return generate_token_range_display_html(state.current_tokens, None, None, 0), 0


def update_single_token_display(token_selector):
    """Update Single Token display based on dropdown selection."""
    if state.current_tokens is None or len(state.current_tokens) == 0:
        return "<p style='color: gray;'>No tokens available</p>"
    
    if token_selector is None or token_selector == "":
        return generate_single_token_display_html(state.current_tokens, None)
    
    try:
        # Extract token index from selector string
        token_idx = int(token_selector.split(":")[0].split()[-1])
        return generate_single_token_display_html(state.current_tokens, token_idx)
    except:
        return generate_single_token_display_html(state.current_tokens, None)


def visualize_attention_unified(
    selection_mode: str,
    token_index: int,
    token_range_start_idx: int,
    token_range_end_idx: int,
    aggregation_method: str,
    colormap: str,
    alpha: float
) -> Optional[Dict]:
    """
    Unified visualization function that handles both single token and token range.
    Returns attention maps for all 28 layers + 1 mean map.
    
    Args:
        selection_mode: "Single Token" or "Token Range"
        token_index: Token index (for single token mode)
        token_range_start_idx: Start token index (for range mode)
        token_range_end_idx: End token index (for range mode)
        aggregation_method: How to aggregate attention heads (mean/max/min)
        colormap: Colormap name
        alpha: Overlay transparency
        
    Returns:
        Dictionary mapping layer names to attention heatmaps
    """
    global current_attention_maps
    
    if selection_mode == "Single Token":
        # Use single token visualization with index
        if state.current_tokens is None or len(state.current_tokens) == 0:
            print("ERROR: No tokens available")
            return (None, None, None, None, 
                    gr.update(), gr.update(), gr.update(), gr.update(), 
                    "<div style='color: gray;'>No data</div>")
        
        # Create fake token_selector string for compatibility with existing function
        token_idx = int(token_index) if token_index is not None else 0
        token_idx = max(0, min(token_idx, len(state.current_tokens) - 1))
        token_selector = f"Token {token_idx}: '{state.current_tokens[token_idx]}'"
        
        attention_maps = visualize_token_attention(
            token_selector=token_selector,
            aggregation_method=aggregation_method,
            colormap=colormap,
            alpha=alpha
        )
    else:
        # Use token range visualization with provided indices
        try:
            # Check if indices are valid (not None)
            if token_range_start_idx is None or token_range_end_idx is None:
                print("ERROR: Please select both start and end indices by clicking on tokens")
                return (None, None, None, None, 
                        gr.update(), gr.update(), gr.update(), gr.update(), 
                        "<div style='color: gray;'>No data</div>")
            
            # Handle empty string values
            if token_range_start_idx == '' or token_range_end_idx == '':
                print("ERROR: Please select both start and end indices by clicking on tokens")
                return (None, None, None, None, 
                        gr.update(), gr.update(), gr.update(), gr.update(), 
                        "<div style='color: gray;'>No data</div>")
            
            token_start_idx = int(token_range_start_idx)
            token_end_idx = int(token_range_end_idx)
            
            if token_start_idx > token_end_idx:
                print("WARNING: Start token is after end token, swapping them")
                token_start_idx, token_end_idx = token_end_idx, token_start_idx
            
            # Use token range visualization
            attention_maps = visualize_token_range_attention(
                token_start_idx=token_start_idx,
                token_end_idx=token_end_idx,
                aggregation_method=aggregation_method,
                colormap=colormap,
                alpha=alpha
            )
        except Exception as e:
            print(f"ERROR: Failed to visualize token range: {e}")
            import traceback
            traceback.print_exc()
            return (None, None, None, None, 
                    gr.update(), gr.update(), gr.update(), gr.update(), 
                    "<div style='color: gray;'>No data</div>")
    
    if attention_maps is None:
        return (None, None, None, None, 
                gr.update(), gr.update(), gr.update(), gr.update(), 
                "<div style='color: gray;'>No data</div>")
    
    # Store attention maps globally
    current_attention_maps = attention_maps
    
    # Calculate attention distribution (image vs prompt text)
    # Note: Always use "mean" aggregation for distribution calculation to ensure
    # consistent results regardless of visualization aggregation method choice
    attention_dist = None
    if selection_mode == "Single Token":
        # For single token
        token_idx = int(token_index) if token_index is not None else 0
        token_idx = max(0, min(token_idx, len(state.current_tokens) - 1))
        
        input_length = state.current_input_ids.shape[1]
        absolute_token_position = input_length + token_idx
        step_key = absolute_token_position
        
        if step_key in state.current_attention:
            step_attention = state.current_attention[step_key]
            available_layers = sorted(step_attention.keys())
            sample_layer = available_layers[0] if available_layers else None
            available_heads = sorted(step_attention[sample_layer].keys()) if sample_layer is not None else []
            
            # Always use "mean" for attention distribution calculation (independent of visualization aggregation)
            attention_dist = state.current_processor.get_attention_distribution(
                step_attention,
                token_position=absolute_token_position,
                input_length=input_length,
                layer_indices=available_layers,
                head_indices=available_heads,
                aggregation_method="mean"
            )
    else:
        # For token range - calculate average distribution
        input_length = state.current_input_ids.shape[1]
        available_steps = sorted(state.current_attention.keys())
        
        distributions = []
        for token_idx in range(int(token_range_start_idx), int(token_range_end_idx) + 1):
            if token_idx >= len(state.current_tokens):
                continue
            
            absolute_token_position = input_length + token_idx
            step_key = absolute_token_position
            
            if step_key not in state.current_attention:
                if not available_steps:
                    continue
                step_key = min(available_steps, key=lambda x: abs(x - step_key))
            
            step_attention = state.current_attention[step_key]
            available_layers = sorted(step_attention.keys())
            sample_layer = available_layers[0] if available_layers else None
            available_heads = sorted(step_attention[sample_layer].keys()) if sample_layer is not None else []
            
            # Always use "mean" for attention distribution calculation (independent of visualization aggregation)
            dist = state.current_processor.get_attention_distribution(
                step_attention,
                token_position=absolute_token_position,
                input_length=input_length,
                layer_indices=available_layers,
                head_indices=available_heads,
                aggregation_method="mean"
            )
            if dist:
                distributions.append(dist)
        
        # Average the distributions
        if distributions:
            avg_image = sum(d['image_percentage'] for d in distributions) / len(distributions)
            avg_prompt_text = sum(d['prompt_text_percentage'] for d in distributions) / len(distributions)
            attention_dist = {
                'image_percentage': avg_image,
                'prompt_text_percentage': avg_prompt_text
            }
    
    # Format attention distribution for display
    if attention_dist:
        dist_html = f"""
        <div style="padding: 10px; background: #f5f5f5; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;">
            <div style="font-size: 14px; color: #333; line-height: 1.8;">
                <strong>Attention Distribution:</strong><br>
                Image: <span style="color: #1976d2; font-weight: bold;">{attention_dist['image_percentage']:.1f}%</span> | 
                Prompt Text: <span style="color: #388e3c; font-weight: bold;">{attention_dist['prompt_text_percentage']:.1f}%</span>
            </div>
        </div>
        """
    else:
        dist_html = "<div style='padding: 10px; color: gray;'>No attention distribution available</div>"
    
    # Get layer choices
    layer_choices = sorted([k for k in attention_maps.keys() if k.startswith("Layer")], 
                          key=lambda x: int(x.split()[1]))
    layer_choices.append("Mean (All Layers)")
    
    # Create default visualizations for 4 frames
    # Frame 1 (top-left): Mean, Frame 2 (top-right): Layer 0
    # Frame 3 (bottom-left): Layer 14, Frame 4 (bottom-right): Layer 27
    visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)
    
    default_layers = [
        "Mean (All Layers)",  # Frame 1 - top left
        "Layer 0",            # Frame 2 - top right
        "Layer 14",           # Frame 3 - bottom left
        "Layer 27"            # Frame 4 - bottom right
    ]
    
    images = []
    for layer_name in default_layers:
        if layer_name in attention_maps:
            img = visualizer.create_heatmap_overlay(
                state.current_image,
                attention_maps[layer_name]
            )
            images.append(img)
        else:
            images.append(None)
    
    # Update dropdowns with layer choices and set default values
    dropdown_updates = [gr.update(choices=layer_choices, value=default_layers[i]) for i in range(4)]
    
    return (images[0], images[1], images[2], images[3],
            dropdown_updates[0], dropdown_updates[1], dropdown_updates[2], dropdown_updates[3],
            dist_html)


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
                model_path_input = gr.Textbox(
                    label="Model Path",
                    placeholder="Enter model path or Hugging Face model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)",
                    value=config.MODEL_NAME,
                    lines=1,
                    info="Local path or Hugging Face model ID"
                )
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
                    temperature = gr.Slider(0.0, 2.0, value=0.1, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Top-p")

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
                    value="Token Range",
                    label="Selection Mode",
                    info="Choose to visualize a single token or aggregate attention over multiple tokens"
                )
                
                # Single token selector (simple slider)
                token_index = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=1,
                    label="Token Index",
                    interactive=True,
                    visible=False
                )
                
                # Token range selectors (two sliders)
                with gr.Row(visible=True) as token_range_row:
                    token_range_start = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="Start Index",
                        interactive=True
                    )
                    token_range_end = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=4,
                        step=1,
                        label="End Index",
                        interactive=True
                    )
                
                # Token display (for both modes)
                token_display = gr.HTML(
                    value="<p style='color: gray;'>Generate text first to see tokens</p>"
                )

                with gr.Accordion("Attention Settings", open=True):
                    aggregation = gr.Radio(
                        ["Mean", "Max", "Min"],
                        value="Mean",
                        label="Head Aggregation Method",
                        info="How to aggregate attention across multiple heads"
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

                visualize_btn = gr.Button("Visualize", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 🎨 Attention Visualization (2×2 Grid)")
                
                # Attention distribution display
                attention_dist_display = gr.HTML(
                    value="<div style='padding: 10px; color: gray; text-align: center;'>Click 'Visualize' to see attention distribution</div>",
                    label=None
                )
                
                with gr.Row():
                    with gr.Column():
                        layer_select_1 = gr.Dropdown(
                            choices=[],
                            label="Select Layer for Frame 1",
                            value=None,
                            interactive=True
                        )
                        output_viz_1 = gr.Image(label="Frame 1", type="pil", height=300)
                    
                    with gr.Column():
                        layer_select_2 = gr.Dropdown(
                            choices=[],
                            label="Select Layer for Frame 2",
                            value=None,
                            interactive=True
                        )
                        output_viz_2 = gr.Image(label="Frame 2", type="pil", height=300)
                
                with gr.Row():
                    with gr.Column():
                        layer_select_3 = gr.Dropdown(
                            choices=[],
                            label="Select Layer for Frame 3",
                            value=None,
                            interactive=True
                        )
                        output_viz_3 = gr.Image(label="Frame 3", type="pil", height=300)
                    
                    with gr.Column():
                        layer_select_4 = gr.Dropdown(
                            choices=[],
                            label="Select Layer for Frame 4",
                            value=None,
                            interactive=True
                        )
                        output_viz_4 = gr.Image(label="Frame 4", type="pil", height=300)


        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=[model_path_input],
            outputs=load_status
        )

        generate_btn.click(
            fn=generate_with_attention,
            inputs=[image_input, prompt_input, max_tokens, temperature, top_p],
            outputs=[output_text, gen_status,
                    token_display,
                    token_index, token_range_start, token_range_end]
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
            outputs=[token_index, token_range_row]
        )
        
        # Update display when single token index changes
        def update_single_display(idx):
            if state.current_tokens is None:
                return "<p style='color: gray;'>No tokens available</p>"
            token_idx = int(idx) if idx is not None else 0
            return generate_single_token_display_html(state.current_tokens, token_idx)
        
        token_index.change(
            fn=update_single_display,
            inputs=[token_index],
            outputs=[token_display]
        )
        
        # Update display when range sliders change
        def update_range_display(start, end):
            if state.current_tokens is None:
                return "<p style='color: gray;'>No tokens available</p>"
            start_idx = int(start) if start is not None else 0
            end_idx = int(end) if end is not None else 0
            return generate_token_range_display_html(state.current_tokens, start_idx, end_idx, 2)
        
        token_range_start.change(
            fn=update_range_display,
            inputs=[token_range_start, token_range_end],
            outputs=[token_display]
        )
        
        token_range_end.change(
            fn=update_range_display,
            inputs=[token_range_start, token_range_end],
            outputs=[token_display]
        )

        visualize_btn.click(
            fn=visualize_attention_unified,
            inputs=[
                selection_mode, token_index, 
                token_range_start, token_range_end,
                aggregation, colormap, alpha_slider
            ],
            outputs=[
                output_viz_1, output_viz_2, output_viz_3, output_viz_4,
                layer_select_1, layer_select_2, layer_select_3, layer_select_4,
                attention_dist_display
            ]
        )
        
        # Add event handlers for layer selection dropdowns
        def update_frame_1(layer_name, cmap, a):
            return create_layer_visualization(layer_name, cmap, a)
        
        def update_frame_2(layer_name, cmap, a):
            return create_layer_visualization(layer_name, cmap, a)
        
        def update_frame_3(layer_name, cmap, a):
            return create_layer_visualization(layer_name, cmap, a)
        
        def update_frame_4(layer_name, cmap, a):
            return create_layer_visualization(layer_name, cmap, a)
        
        layer_select_1.change(
            fn=update_frame_1,
            inputs=[layer_select_1, colormap, alpha_slider],
            outputs=output_viz_1
        )
        
        layer_select_2.change(
            fn=update_frame_2,
            inputs=[layer_select_2, colormap, alpha_slider],
            outputs=output_viz_2
        )
        
        layer_select_3.change(
            fn=update_frame_3,
            inputs=[layer_select_3, colormap, alpha_slider],
            outputs=output_viz_3
        )
        
        layer_select_4.change(
            fn=update_frame_4,
            inputs=[layer_select_4, colormap, alpha_slider],
            outputs=output_viz_4
        )
        
        # Update all frames when colormap or alpha changes
        def update_all_frames(cmap, a, l1, l2, l3, l4):
            return (
                create_layer_visualization(l1, cmap, a) if l1 else None,
                create_layer_visualization(l2, cmap, a) if l2 else None,
                create_layer_visualization(l3, cmap, a) if l3 else None,
                create_layer_visualization(l4, cmap, a) if l4 else None
            )
        
        colormap.change(
            fn=update_all_frames,
            inputs=[colormap, alpha_slider, layer_select_1, layer_select_2, layer_select_3, layer_select_4],
            outputs=[output_viz_1, output_viz_2, output_viz_3, output_viz_4]
        )
        
        alpha_slider.change(
            fn=update_all_frames,
            inputs=[colormap, alpha_slider, layer_select_1, layer_select_2, layer_select_3, layer_select_4],
            outputs=[output_viz_1, output_viz_2, output_viz_3, output_viz_4]
        )

    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Qwen2.5-VL Attention Visualization Tool")
    print("="*60)
    print(f"Default Model: {config.MODEL_NAME}")
    print(f"Device: {config.MODEL_DEVICE_MAP}")
    print(f"Server: {config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}")
    print("="*60)
    print("Note: You can change the model path in the web interface.")

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE
    )
