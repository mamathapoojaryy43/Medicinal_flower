"""
build_model.py
==============
Medicinal Flower Recognition Using Deep Learning and Explainable AI.

This script defines the MobileNetV2 transfer learning model architecture,
detects the 31 class labels from the dataset directory, saves the class
mapping to models/class_names.json, and prints the model architecture summary.
"""

import json
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dynamic project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLASS_MAPPING_FILE = os.path.join(MODELS_DIR, "class_names.json")

# Model configuration hyperparameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 0.001

def detect_and_save_classes(train_dir, output_json_path):
    """
    Scans the training dataset folder to detect class subfolders,
    builds class-to-index and index-to-class mappings, and saves them to JSON.
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training dataset directory not found: {train_dir}")

    # 1. Discover all class folders and sort alphabetically
    classes = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    classes.sort()

    total_classes = len(classes)
    print("========================================")
    print("       DETECTED DATASET CLASSES         ")
    print("========================================")
    print(f"Dataset path: {train_dir}")
    print(f"Total classes found: {total_classes}\n")

    # 2. Build mapping dictionaries
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {str(idx): class_name for idx, class_name in enumerate(classes)}

    # 3. Print mapping table
    print(f"{'Index':<6} | {'Class Name':<30}")
    print("-" * 40)
    for class_name, idx in class_to_idx.items():
        print(f"{idx:<6} | {class_name:<30}")
    print("-" * 40 + "\n")

    # 4. Save to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    mapping_data = {
        "total_classes": total_classes,
        "class_names": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=4)

    print(f"[OK] Class mapping successfully saved to: {output_json_path}\n")
    return classes, class_to_idx

def build_mobilenetv2_model(num_classes=31, input_shape=(224, 224, 3), freeze_base=True):
    """
    Creates and compiles a MobileNetV2 transfer learning model.

    Parameters:
        num_classes (int): Number of target classification classes (default: 31).
        input_shape (tuple): Dimensions of input RGB images (default: (224, 224, 3)).
        freeze_base (bool): Whether to freeze the pretrained MobileNetV2 feature extractor.

    Returns:
        tf.keras.Model: Compiled Keras classification model.
    """
    # 1. Define Input Layer (224x224 RGB image tensor)
    inputs = Input(shape=input_shape, name="input_image")

    # 2. Preprocess input images for MobileNetV2 (scales pixels to [-1, 1])
    x = preprocess_input(inputs)

    # 3. Load pretrained MobileNetV2 base (ImageNet weights, excluding top classification layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # 4. Freeze or unfreeze base model feature extraction layers
    base_model.trainable = not freeze_base

    # Pass preprocessed tensor through feature extractor
    x = base_model(x, training=False)

    # 5. Custom Classification Head
    # Global Average Pooling reduces feature maps (7x7x1280) to a 1D vector (1280)
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = Dropout(0.3, name="dropout_1")(x)

    # Fully-connected dense layer with ReLU activation
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.3, name="dropout_2")(x)

    # Final Output Layer with Softmax activation for 31-class probabilities
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    # 6. Construct Model
    model = Model(inputs=inputs, outputs=outputs, name="Medicinal_Flower_MobileNetV2")

    # 7. Compile Model with Adam optimizer and Categorical Crossentropy Loss
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    print("========================================")
    print("   MEDICINAL FLOWER MODEL CREATION     ")
    print("========================================")

    # 1. Detect dataset classes and generate class_names.json
    classes, class_to_idx = detect_and_save_classes(DATASET_DIR, CLASS_MAPPING_FILE)
    num_classes = len(classes)

    # 2. Build MobileNetV2 model architecture
    print("Building MobileNetV2 model architecture...")
    model = build_mobilenetv2_model(
        num_classes=num_classes,
        input_shape=INPUT_SHAPE,
        freeze_base=True
    )

    # 3. Print Complete Keras Model Architecture Summary
    print("\n========================================")
    print("        MODEL ARCHITECTURE SUMMARY       ")
    print("========================================")
    model.summary()
    print("========================================\n")
    print(f"[OK] Model successfully built with {num_classes} output classes.")
    print("[OK] Base MobileNetV2 layers are frozen (trainable = False).")
    print(f"[OK] Class names JSON saved at: {CLASS_MAPPING_FILE}")

if __name__ == "__main__":
    main()
