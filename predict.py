"""
Medicinal Flower Recognition & Explainable AI (Grad-CAM)
=========================================================
CLI Inference script for single-image classification, confidence
thresholding (out-of-distribution rejection), and visual explainability.

Usage examples:
    python predict.py Hibiscus_flower_TZ.jpg
    python predict.py --image crownflower.jpeg --gradcam
    python predict.py --image rose.jpg --output_json result.json
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# Reconfigure stdout to UTF-8 for reliable console rendering across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Determine project root dynamically from file location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "medicinal_flower_model.keras")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_names.json")
CONFIG_PATH = os.path.join(MODELS_DIR, "prediction_config.json")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Classify medicinal flower images with optional Grad-CAM explainability."
    )
    # Support both positional argument and --image flag
    parser.add_argument(
        "positional_image",
        nargs="?",
        default=None,
        help="Path to target flower image (positional)"
    )
    parser.add_argument(
        "--image", "-i",
        dest="flag_image",
        default=None,
        help="Path to target flower image"
    )
    parser.add_argument(
        "--gradcam", "-g",
        action="store_true",
        help="Generate and save a Grad-CAM visual explanation panel"
    )
    parser.add_argument(
        "--gradcam_out",
        default=None,
        help="Custom destination filepath for the Grad-CAM visualization image"
    )
    parser.add_argument(
        "--output_json", "-o",
        default=None,
        help="Save prediction results payload to a JSON file"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Custom confidence threshold (0.0 - 1.0) for out-of-distribution rejection"
    )
    return parser.parse_args()


def load_model_and_config(custom_threshold=None):
    """Load model checkpoint, target class labels, and inference thresholds."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
    if not os.path.exists(CLASS_MAPPING_PATH):
        raise FileNotFoundError(f"Class mapping file not found at: {CLASS_MAPPING_PATH}")

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    class_names = class_info["class_names"]

    threshold = 0.60
    unrecognized_msg = "This flower could not be confidently matched to any of the 31 supported classes."

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            threshold = cfg.get("confidence_threshold", threshold)
            unrecognized_msg = cfg.get("unrecognized_message", unrecognized_msg)

    if custom_threshold is not None:
        threshold = custom_threshold

    # Suppress verbose TensorFlow warnings for clean CLI output
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, class_names, threshold, unrecognized_msg


def preprocess_image(image_path):
    """Load, resize to 224x224 RGB, and format image tensor."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((224, 224))
    rgb_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(rgb_array, axis=0)  # Shape: (1, 224, 224, 3)
    return rgb_array, img_batch


def generate_gradcam_figure(model, img_raw_rgb, top_pred_idx, predicted_class, confidence_pct, output_path):
    """Generate 3-panel Grad-CAM visualization using pre-softmax logits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    # Base feature extractor
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 base layer not found in target model.")

    conv_layer_name = "out_relu"
    try:
        conv_layer = base_model.get_layer(conv_layer_name)
    except ValueError:
        conv_layer_name = "Conv_1"
        conv_layer = base_model.get_layer(conv_layer_name)

    base_sub_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output]
    )

    img_batch_raw = np.expand_dims(img_raw_rgb, axis=0)
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(np.copy(img_batch_raw))

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = base_sub_model(preprocessed_img)
        tape.watch(conv_outputs)

        x = model.get_layer("global_avg_pool")(base_outputs)
        x = model.get_layer("batch_norm")(x)
        x = model.get_layer("dropout_1")(x, training=False)
        x = model.get_layer("dense_256")(x)
        x = model.get_layer("dropout_2")(x, training=False)

        # Pre-softmax logits prevent gradient saturation
        predictions_layer = model.get_layer("predictions")
        kernel = predictions_layer.kernel
        bias = predictions_layer.bias
        pre_softmax_logits = tf.matmul(x, kernel) + bias

        loss = pre_softmax_logits[:, top_pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match image dimensions
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize((224, 224), resample=Image.BICUBIC)
    heatmap_resized = np.array(heatmap_img) / 255.0

    cmap = plt.cm.jet
    color_heatmap = cmap(heatmap_resized)[:, :, :3]

    orig_norm = img_raw_rgb / 255.0
    overlay = 0.6 * orig_norm + 0.4 * color_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    axes[0].imshow(orig_norm)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap ('{conv_layer_name}')", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Explanation: {predicted_class} ({confidence_pct:.2f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_arguments()
    target_image = args.flag_image or args.positional_image

    if not target_image:
        print("[ERROR] No image path provided.")
        print("Usage:")
        print("  python predict.py <path/to/image.jpg>")
        print("  python predict.py --image <path/to/image.jpg> --gradcam --output_json result.json")
        sys.exit(1)

    image_path = target_image.strip("\"'")
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    model, class_names, threshold, unrecognized_msg = load_model_and_config(args.threshold)
    rgb_array, img_batch = preprocess_image(image_path)

    # Run neural inference
    predictions = model.predict(img_batch, verbose=0)[0]
    max_prob = float(np.max(predictions))
    top_pred_idx = int(np.argmax(predictions))
    predicted_class = class_names[top_pred_idx]
    confidence_pct = max_prob * 100.0

    top_5_indices = np.argsort(predictions)[::-1][:5]
    top_predictions = [
        {"class_name": class_names[idx], "probability": round(float(predictions[idx] * 100.0), 2)}
        for idx in top_5_indices
    ]

    is_recognized = max_prob >= threshold

    # Print formatted output
    print("\n" + "=" * 50)
    print("      MEDICINAL FLOWER PREDICTION REPORT      ")
    print("=" * 50)
    print(f"Target Image : {os.path.basename(image_path)}")

    if is_recognized:
        print("Status       : Recognized Supported Flower")
        print(f"Prediction   : {predicted_class}")
        print(f"Confidence   : {confidence_pct:.2f}% (Threshold: {threshold * 100:.1f}%)")
    else:
        print("Status       : Not Recognized / Unsupported")
        print(f"Confidence   : {confidence_pct:.2f}% (Below threshold of {threshold * 100:.1f}%)")
        print(f"Notice       : {unrecognized_msg}")

    print("\nTop 5 Candidates:")
    for rank, item in enumerate(top_predictions, start=1):
        print(f"  {rank}. {item['class_name']:<22} : {item['probability']:>6.2f}%")

    gradcam_saved_path = None
    if args.gradcam:
        default_out = f"gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        out_path = args.gradcam_out or os.path.join(PROJECT_ROOT, default_out)
        try:
            gradcam_saved_path = generate_gradcam_figure(
                model, rgb_array, top_pred_idx, predicted_class, confidence_pct, out_path
            )
            print(f"\nGrad-CAM Saved: {gradcam_saved_path}")
        except Exception as e:
            print(f"\n[WARNING] Could not generate Grad-CAM visualization: {e}")

    print("=" * 50 + "\n")

    # Export JSON if requested
    if args.output_json:
        result_payload = {
            "image_path": os.path.abspath(image_path),
            "recognized": is_recognized,
            "predicted_class": predicted_class if is_recognized else None,
            "confidence_pct": round(confidence_pct, 2),
            "threshold_pct": round(threshold * 100.0, 2),
            "top_predictions": top_predictions,
            "gradcam_image_path": gradcam_saved_path
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=4)
        print(f"[OK] Prediction results exported to: {args.output_json}\n")


if __name__ == "__main__":
    main()
