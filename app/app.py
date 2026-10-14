import os
import sys
import json
import uuid
import numpy as np
from PIL import Image

# Use non-interactive Matplotlib backend for thread-safe server visualization rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf

# Suppress verbose TensorFlow CPP logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Initialize Flask App
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Define Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))

UPLOAD_FOLDER = os.path.join(APP_DIR, 'static', 'uploads')
GRADCAM_FOLDER = os.path.join(APP_DIR, 'static', 'gradcam')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Global Model & Data Storage (Loaded ONCE at startup)
MODEL = None
CLASS_NAMES = []
PREDICTION_CONFIG = {}
FLOWER_INFO_DB = {}
BASE_SUB_MODEL = None
LAST_CONV_LAYER_NAME = "out_relu"

def allowed_file(filename):
    """Check if uploaded filename has a supported image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_resources():
    """Load trained Keras model, class mapping, config, and flower information ONCE on startup."""
    global MODEL, CLASS_NAMES, PREDICTION_CONFIG, FLOWER_INFO_DB, BASE_SUB_MODEL, LAST_CONV_LAYER_NAME
    
    model_path = os.path.join(PROJECT_ROOT, "models", "medicinal_flower_model.keras")
    class_mapping_path = os.path.join(PROJECT_ROOT, "models", "class_names.json")
    config_path = os.path.join(PROJECT_ROOT, "models", "prediction_config.json")
    info_db_path = os.path.join(APP_DIR, "flower_information.json")
    
    print("\n=======================================================")
    print("   MEDICINAL FLOWER RECOGNITION - FLASK BACKEND INIT   ")
    print("=======================================================")
    
    # 1. Load Class Mapping
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at: {class_mapping_path}")
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)["class_names"]
    print(f" -> Loaded {len(CLASS_NAMES)} class names from class_names.json")
    
    # 2. Load Prediction Config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prediction config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        PREDICTION_CONFIG = json.load(f)
    print(f" -> Loaded prediction config (Threshold: {PREDICTION_CONFIG.get('confidence_threshold', 0.60)})")
    
    # 3. Load Flower Information Database
    if not os.path.exists(info_db_path):
        raise FileNotFoundError(f"Flower information DB not found at: {info_db_path}")
    with open(info_db_path, "r", encoding="utf-8") as f:
        FLOWER_INFO_DB = json.load(f)
    print(f" -> Loaded flower information database with {len(FLOWER_INFO_DB)} entries")
    
    # 4. Load Trained Keras Model ONCE
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    print(f" -> Loading Keras model from {model_path}...")
    MODEL = tf.keras.models.load_model(model_path, compile=False)
    print(" -> [OK] Model loaded successfully into memory.")
    
    # 5. Initialize Grad-CAM Sub-Model for feature map extraction
    base_model = None
    for layer in MODEL.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break
            
    if base_model is not None:
        try:
            conv_layer = base_model.get_layer("out_relu")
            LAST_CONV_LAYER_NAME = "out_relu"
        except ValueError:
            conv_layer = base_model.get_layer("Conv_1")
            LAST_CONV_LAYER_NAME = "Conv_1"
            
        BASE_SUB_MODEL = tf.keras.Model(
            inputs=base_model.input,
            outputs=[conv_layer.output, base_model.output]
        )
        print(f" -> [OK] Grad-CAM Sub-Model initialized with layer '{LAST_CONV_LAYER_NAME}'")
    print("=======================================================\n")

# Load resources when module is imported/executed
load_resources()

def generate_gradcam_visualization(img_batch_raw, top_pred_idx, predicted_class, confidence_pct, output_filename):
    """
    Computes Grad-CAM gradients and saves a 3-panel visualization (Original, Heatmap, Overlay)
    to app/static/gradcam/<output_filename>.
    """
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(np.copy(img_batch_raw))
    
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = BASE_SUB_MODEL(preprocessed_img)
        tape.watch(conv_outputs)
        
        x = MODEL.get_layer("global_avg_pool")(base_outputs)
        x = MODEL.get_layer("batch_norm")(x)
        x = MODEL.get_layer("dropout_1")(x, training=False)
        x = MODEL.get_layer("dense_256")(x)
        x = MODEL.get_layer("dropout_2")(x, training=False)
        
        predictions_layer = MODEL.get_layer("predictions")
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
    
    # Resize heatmap to 224x224
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize((224, 224), resample=Image.BICUBIC)
    heatmap_resized = np.array(heatmap_img) / 255.0
    
    cmap = plt.cm.jet
    color_heatmap = cmap(heatmap_resized)[:, :, :3]
    
    orig_norm = img_batch_raw[0] / 255.0
    overlay = 0.6 * orig_norm + 0.4 * color_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    axes[0].imshow(orig_norm)
    axes[0].set_title("Uploaded Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap ('{LAST_CONV_LAYER_NAME}')", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Explanation: {predicted_class} ({confidence_pct:.2f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    output_path = os.path.join(GRADCAM_FOLDER, output_filename)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    
    return f"/static/gradcam/{output_filename}"

@app.route('/', methods=['GET'])
def index():
    """GET / endpoint - Root health check and API status."""
    templates_dir = os.path.join(APP_DIR, 'templates')
    if os.path.exists(os.path.join(templates_dir, 'index.html')):
        return render_template('index.html')
    return jsonify({
        "status": "online",
        "service": "Medicinal Flower Recognition API",
        "model_loaded": MODEL is not None,
        "total_classes": len(CLASS_NAMES),
        "confidence_threshold": PREDICTION_CONFIG.get("confidence_threshold", 0.60)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """POST /predict endpoint - Classifies uploaded flower image and returns Explainable AI Grad-CAM visualization."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part provided in the request."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Please upload a valid JPG, JPEG, or PNG image."}), 400
        
    try:
        filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex[:8]
        saved_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(filepath)
        
        # Load and Preprocess Image
        orig_img = Image.open(filepath).convert("RGB")
        resized_img = orig_img.resize((224, 224))
        orig_rgb_array = np.array(resized_img, dtype=np.float32)
        img_batch_raw = np.expand_dims(orig_rgb_array, axis=0)  # Shape: (1, 224, 224, 3)
        
        # Run Model Prediction
        predictions = MODEL.predict(img_batch_raw, verbose=0)[0]
        max_prob = float(np.max(predictions))
        top_pred_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[top_pred_idx]
        confidence_pct = max_prob * 100.0
        
        threshold = float(PREDICTION_CONFIG.get("confidence_threshold", 0.60))
        threshold_pct = threshold * 100.0
        
        # Format Top 5 Predictions
        top_5_indices = np.argsort(predictions)[::-1][:5]
        top_predictions = [
            {
                "class_name": CLASS_NAMES[idx],
                "probability": round(float(predictions[idx] * 100.0), 2)
            }
            for idx in top_5_indices
        ]
        
        uploaded_image_url = f"/static/uploads/{saved_filename}"
        
        # Handle Unknown / Below Threshold Predictions
        if max_prob < threshold:
            return jsonify({
                "recognized": False,
                "message": PREDICTION_CONFIG.get(
                    "unrecognized_message",
                    "This flower could not be confidently matched to any of the 31 supported flower classes."
                ),
                "confidence": round(confidence_pct, 2),
                "threshold": round(threshold_pct, 2),
                "top_predictions": top_predictions,
                "uploaded_image_url": uploaded_image_url
            })
            
        # Recognized Supported Flower
        info = FLOWER_INFO_DB.get(predicted_class, {
            "common_name": predicted_class,
            "scientific_name": "N/A",
            "short_botanical_description": "Information not available.",
            "traditional_uses": "Information not available."
        })
        
        # Generate Grad-CAM Overlay Image
        gradcam_filename = f"gradcam_{saved_filename.rsplit('.', 1)[0]}.png"
        gradcam_image_url = generate_gradcam_visualization(
            img_batch_raw, top_pred_idx, predicted_class, confidence_pct, gradcam_filename
        )
        
        return jsonify({
            "recognized": True,
            "predicted_flower": predicted_class,
            "confidence": round(confidence_pct, 2),
            "common_name": info.get("common_name", predicted_class),
            "scientific_name": info.get("scientific_name", "N/A"),
            "short_botanical_description": info.get("short_botanical_description", ""),
            "traditional_uses": info.get("traditional_uses", ""),
            "top_predictions": top_predictions,
            "uploaded_image_url": uploaded_image_url,
            "gradcam_image_url": gradcam_image_url
        })
        
    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({"error": f"An error occurred while processing the image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
