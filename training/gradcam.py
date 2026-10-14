"""
Medicinal Flower Recognition - Grad-CAM Explainable AI Implementation
----------------------------------------------------------------------
This module implements mathematically valid Grad-CAM (Gradient-weighted Class Activation Mapping)
for MobileNetV2 transfer learning.

Grad-CAM Theory & Math:
----------------------
1. Feature Map Extraction:
   Extract spatial feature maps A^k from the final 4D convolutional activation layer of MobileNetV2 ('out_relu', shape: 7x7x1280).

2. Un-Saturated Gradient Computation (Pre-Softmax Logits):
   Compute gradients of the un-normalized pre-softmax score z^c for predicted class c:
   
       gradient = d(z^c) / d(A^k)
       
   Note: Using pre-softmax logits z^c instead of post-softmax probability S_c avoids Softmax saturation
   (where S_c -> 1.0 causes d(S_c)/d(z_c) = S_c(1 - S_c) -> 0.0), preserving strong, meaningful gradients.

3. Global Average Pooling Weights:
   alpha_k^c = (1 / Z) * sum_i sum_j ( d(z^c) / d(A^i,j^k) )

4. Weighted Combination & ReLU:
   L_Grad-CAM^c = ReLU( sum_k ( alpha_k^c * A^k ) )

5. Normalization & Visualization Overlay:
   Normalize heatmaps between [0, 1] and superimpose jet colormap over the original RGB image.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def load_gradcam_submodel(model, conv_layer_name="out_relu"):
    """
    Constructs a sub-model that outputs both the target convolutional feature maps
    and the final MobileNetV2 base model outputs.
    """
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break
            
    if base_model is None:
        raise ValueError("MobileNetV2 base layer not found in target model.")
        
    try:
        conv_layer = base_model.get_layer(conv_layer_name)
    except ValueError:
        raise ValueError(f"Convolutional layer '{conv_layer_name}' not found in MobileNetV2 base model.")
        
    base_sub_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output]
    )
    return base_sub_model


def generate_gradcam_heatmap(model, preprocessed_img_batch, top_pred_idx, conv_layer_name="out_relu"):
    """
    Generates a normalized 2D Grad-CAM heatmap for a target class using pre-softmax logits.
    
    Returns:
    --------
    heatmap_norm : 2D numpy array (values in [0, 1])
    debug_stats  : dict of numerical diagnostics
    """
    base_sub_model = load_gradcam_submodel(model, conv_layer_name=conv_layer_name)
    
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = base_sub_model(preprocessed_img_batch)
        tape.watch(conv_outputs)
        
        # Pass base_outputs through classification head layers up to dense_256
        x = model.get_layer("global_avg_pool")(base_outputs)
        x = model.get_layer("batch_norm")(x)
        x = model.get_layer("dropout_1")(x, training=False)
        x = model.get_layer("dense_256")(x)
        x = model.get_layer("dropout_2")(x, training=False)
        
        # Compute PRE-SOFTMAX logit for top_pred_idx to avoid Softmax saturation
        predictions_layer = model.get_layer("predictions")
        kernel = predictions_layer.kernel
        bias = predictions_layer.bias
        pre_softmax_logits = tf.matmul(x, kernel) + bias
        
        score = pre_softmax_logits[:, top_pred_idx]
        
    # Calculate gradients of score with respect to feature maps
    grads = tape.gradient(score, conv_outputs)
    
    # Global average pool gradients over spatial dimensions (H, W)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by pooled gradients
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU activation to keep only positive influence features
    heatmap = tf.maximum(heatmap, 0)
    
    # Extract numerical debug statistics
    min_val = float(tf.reduce_min(heatmap))
    max_val = float(tf.reduce_max(heatmap))
    mean_val = float(tf.reduce_mean(heatmap))
    grad_min = float(tf.reduce_min(grads))
    grad_max = float(tf.reduce_max(grads))
    
    # Normalize between 0 and 1
    if max_val > 0:
        heatmap_norm = heatmap / max_val
    else:
        heatmap_norm = heatmap
        
    debug_stats = {
        "conv_layer": conv_layer_name,
        "feature_shape": list(conv_outputs.shape),
        "grad_shape": list(grads.shape),
        "grad_min": grad_min,
        "grad_max": grad_max,
        "heatmap_min": min_val,
        "heatmap_max": max_val,
        "heatmap_mean": mean_val
    }
    
    return heatmap_norm.numpy(), debug_stats


def create_gradcam_overlay(orig_img_rgb, heatmap, alpha=0.4):
    """
    Overlays a 2D heatmap onto the original RGB image.
    """
    img_h, img_w = orig_img_rgb.shape[:2]
    
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((img_w, img_h), resample=Image.BICUBIC)
    heatmap_resized = np.array(heatmap_pil) / 255.0
    
    cmap = plt.cm.jet
    color_heatmap = cmap(heatmap_resized)[:, :, :3]
    
    orig_norm = orig_img_rgb.astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * orig_norm + alpha * color_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    
    return orig_norm, heatmap_resized, overlay
