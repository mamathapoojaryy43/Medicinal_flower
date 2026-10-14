import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

def test_gradcam_fix():
    print("=" * 80)
    print("      GRAD-CAM DEBUG & VALIDATION (PRE-SOFTMAX LOGITS)      ")
    print("=" * 80)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "medicinal_flower_model.keras")
    class_mapping_path = os.path.join(project_root, "models", "class_names.json")
    test_dir = os.path.join(project_root, "dataset", "test")
    
    output_dir = os.path.join(project_root, "training", "results", "gradcam_examples")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    class_names = class_info["class_names"]
    
    print("\nLoading Keras model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Identify MobileNetV2 base model
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break
            
    if base_model is None:
        raise ValueError("MobileNetV2 base model not found.")
        
    # We will test both 'out_relu' and 'Conv_1' to compare activations
    selected_layer_name = "out_relu"
    conv_layer = base_model.get_layer(selected_layer_name)
    
    base_sub_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output]
    )
    
    # Pick 5 distinct test classes
    test_classes = ["Hibiscus", "Rose", "Sunflower", "Datura", "Allium_Cepa"]
    
    print("\n" + "=" * 80)
    print(f"Selected Grad-CAM Layer: '{selected_layer_name}'")
    print("=" * 80)
    
    for cls in test_classes:
        cls_dir = os.path.join(test_dir, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if not imgs:
            continue
            
        img_name = imgs[0]
        img_path = os.path.join(cls_dir, img_name)
        
        # Load & Preprocess
        orig_img = Image.open(img_path).convert("RGB")
        resized_img = orig_img.resize((224, 224))
        orig_rgb = np.array(resized_img, dtype=np.float32)
        img_batch_raw = np.expand_dims(orig_rgb, axis=0)
        
        # Prediction
        preds = model.predict(img_batch_raw, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        pred_cls = class_names[top_idx]
        conf_pct = float(preds[top_idx] * 100.0)
        
        # Preprocess copy for MobileNetV2 base model
        preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(np.copy(img_batch_raw))
        
        # GradientTape tracking PRE-SOFTMAX logit
        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = base_sub_model(preprocessed_img)
            tape.watch(conv_outputs)
            
            # Pass base_outputs through head up to dense_256
            x = model.get_layer("global_avg_pool")(base_outputs)
            x = model.get_layer("batch_norm")(x)
            x = model.get_layer("dropout_1")(x, training=False)
            x = model.get_layer("dense_256")(x)
            x = model.get_layer("dropout_2")(x, training=False)
            
            # Compute PRE-SOFTMAX logit for top_idx directly
            pred_layer = model.get_layer("predictions")
            kernel = pred_layer.kernel
            bias = pred_layer.bias
            pre_softmax_logits = tf.matmul(x, kernel) + bias
            
            loss = pre_softmax_logits[:, top_idx]
            
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU & Normalization
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        min_val = tf.math.reduce_min(heatmap)
        mean_val = tf.math.reduce_mean(heatmap)
        
        if max_val > 0:
            heatmap_norm = heatmap / max_val
        else:
            heatmap_norm = heatmap
            
        grad_min = float(tf.reduce_min(grads))
        grad_max = float(tf.reduce_max(grads))
        
        # Print mandatory debugging output
        print(f"\n--- Class: {cls} (Image: {img_name}) ---")
        print(f"Prediction: {pred_cls} ({conf_pct:.2f}%)")
        print(f"Selected Grad-CAM layer: {selected_layer_name}")
        print(f"Feature map shape: {conv_outputs.shape}")
        print(f"Gradient shape: {grads.shape}")
        print(f"Gradient minimum: {grad_min:.8f}")
        print(f"Gradient maximum: {grad_max:.8f}")
        print(f"Heatmap minimum: {float(min_val):.8f}")
        print(f"Heatmap maximum: {float(max_val):.8f}")
        print(f"Heatmap mean: {float(mean_val):.8f}")
        
        # Save images into training/results/gradcam_examples/<ClassName>/
        cls_out_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_out_dir, exist_ok=True)
        
        heatmap_np = heatmap_norm.numpy()
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap_np)).resize((224, 224), resample=Image.BICUBIC)
        heatmap_resized = np.array(heatmap_pil) / 255.0
        
        cmap = plt.cm.jet
        color_heatmap = cmap(heatmap_resized)[:, :, :3]
        
        orig_norm = orig_rgb / 255.0
        overlay = 0.6 * orig_norm + 0.4 * color_heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        
        plt.imsave(os.path.join(cls_out_dir, "original.png"), orig_norm)
        plt.imsave(os.path.join(cls_out_dir, "heatmap.png"), heatmap_resized, cmap="jet")
        plt.imsave(os.path.join(cls_out_dir, "overlay.png"), overlay)
        
        # Also composite panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        axes[0].imshow(orig_norm)
        axes[0].set_title(f"Original: {cls}", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title(f"Raw Heatmap ('{selected_layer_name}')", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Grad-CAM Overlay\n{pred_cls} ({conf_pct:.2f}%)", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(cls_out_dir, "composite_panel.png"), bbox_inches="tight")
        plt.close(fig)
        
    print("\n" + "=" * 80)
    print("Grad-CAM Debug Test Completed.")
    print("=" * 80)

if __name__ == "__main__":
    test_gradcam_fix()
