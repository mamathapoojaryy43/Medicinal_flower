"""
train.py
========
Medicinal Flower Recognition Using Deep Learning and Explainable AI.

This script trains the MobileNetV2 transfer learning model on 31 medicinal flower classes.
It applies data augmentation to the training split, computes balanced class weights,
uses EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks, exports training history,
and plots accuracy/loss visualization curves.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Dynamic Dataset & Model paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(PROJECT_DIR, "dataset", "train")
VAL_DIR = os.path.join(PROJECT_DIR, "dataset", "validation")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "training", "results")

MODEL_PATH = os.path.join(MODELS_DIR, "medicinal_flower_model.keras")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
GRAPHS_PATH = os.path.join(RESULTS_DIR, "training_curves.png")

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
INITIAL_LR = 0.001
FINE_TUNE_LR = 1e-5
RANDOM_SEED = 42

def calculate_class_weights(train_dir, class_names):
    """
    Computes balanced class weights for the 31 classes based on image counts in train_dir.
    Formula: total_samples / (n_classes * class_samples)
    """
    class_counts = []
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        count = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])
        class_counts.append(count)

    total_samples = sum(class_counts)
    n_classes = len(class_names)

    class_weight_dict = {}
    for idx, count in enumerate(class_counts):
        weight = total_samples / (n_classes * count)
        class_weight_dict[idx] = float(weight)

    return class_weight_dict, class_counts, total_samples

def build_data_augmentation():
    """
    Returns a Keras Sequential layer for data augmentation.
    Applied exclusively to the training dataset.
    """
    return tf.keras.Sequential([
        RandomFlip("horizontal", name="aug_horizontal_flip"),
        RandomRotation(0.1, name="aug_rotation"),
        RandomZoom(0.1, name="aug_zoom"),
        RandomTranslation(height_factor=0.1, width_factor=0.1, name="aug_translation")
    ], name="data_augmentation")

def build_mobilenetv2_model(num_classes=31, input_shape=(224, 224, 3)):
    """
    Constructs MobileNetV2 model architecture with frozen base and custom classifier.
    """
    inputs = Input(shape=input_shape, name="input_image")

    # MobileNetV2 specific preprocessing (scales pixels to [-1, 1])
    x = preprocess_input(inputs)

    # Base pretrained model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base initially

    x = base_model(x, training=False)

    # Classification Head
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Medicinal_Flower_MobileNetV2")
    return model, base_model

def main():
    print("========================================")
    print("   MEDICINAL FLOWER MODEL TRAINING      ")
    print("========================================")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data Generators / Datasets
    print("\nLoading Training & Validation Datasets...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    assert num_classes == 31, f"Expected 31 classes, found {num_classes}"

    # 2. Save Class Mapping
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {str(i): c for i, c in enumerate(class_names)}
    mapping_data = {
        "total_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }
    with open(CLASS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=4)
    print(f"[OK] Class names JSON saved to: {CLASS_MAPPING_PATH}")

    # 3. Calculate Class Weights & Image Counts
    class_weight_dict, class_counts, total_train_images = calculate_class_weights(TRAIN_DIR, class_names)
    total_val_images = sum(
        len([f for f in os.listdir(os.path.join(VAL_DIR, c)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
        for c in class_names
    )

    print(f"\nNumber of Training Images   : {total_train_images:,}")
    print(f"Number of Validation Images : {total_val_images:,}")
    print(f"Total Output Classes        : {num_classes}")
    print("\nComputed Class Weights:")
    for cls_name, idx in class_to_idx.items():
        print(f"  - {cls_name:<25}: {class_weight_dict[idx]:.4f} ({class_counts[idx]} images)")

    # 4. Data Pipeline Optimization & Augmentation
    augmentation_model = build_data_augmentation()

    # Apply data augmentation only to train_dataset
    train_ds_augmented = train_dataset.map(
        lambda x, y: (augmentation_model(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 5. Build Model
    model, base_model = build_mobilenetv2_model(num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6. Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )

    callbacks_list = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    # 7. Phase 1 Training (Frozen Base)
    print("\n========================================")
    print(" PHASE 1: Training Classification Head  ")
    print("========================================")
    history_phase1 = model.fit(
        train_ds_augmented,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks_list
    )

    # 8. Phase 2 Fine-Tuning (Unfreeze Top Layers of Base Model)
    print("\n========================================")
    print(" PHASE 2: Fine-Tuning Top Base Layers   ")
    print("========================================")
    base_model.trainable = True

    # Freeze all layers except the last 30 layers
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Total base layers: {len(base_model.layers)}")
    print(f"Fine-tuning starting from layer: {fine_tune_at}")

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    initial_epochs_trained = len(history_phase1.history['loss'])

    history_phase2 = model.fit(
        train_ds_augmented,
        validation_data=val_ds,
        epochs=initial_epochs_trained + FINE_TUNE_EPOCHS,
        initial_epoch=initial_epochs_trained,
        class_weight=class_weight_dict,
        callbacks=callbacks_list
    )

    # 9. Combine Training History
    combined_history = {
        "accuracy": history_phase1.history["accuracy"] + history_phase2.history["accuracy"],
        "val_accuracy": history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"],
        "loss": history_phase1.history["loss"] + history_phase2.history["loss"],
        "val_loss": history_phase1.history["val_loss"] + history_phase2.history["val_loss"],
        "lr": [float(x) for x in (history_phase1.history.get("lr", []) + history_phase2.history.get("lr", []))]
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)
    print(f"\n[OK] Complete training history saved to: {HISTORY_PATH}")

    # 10. Find Best Epoch Stats
    val_accuracies = combined_history["val_accuracy"]
    val_losses = combined_history["val_loss"]
    best_epoch_idx = int(np.argmin(val_losses))  # Best epoch according to val_loss
    best_epoch_num = best_epoch_idx + 1
    best_train_acc = float(combined_history["accuracy"][best_epoch_idx])
    best_val_acc = float(val_accuracies[best_epoch_idx])
    best_val_loss = float(val_losses[best_epoch_idx])

    # 11. Generate Accuracy & Loss Graphs
    epochs_range = range(1, len(combined_history["accuracy"]) + 1)
    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, combined_history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2.5)
    plt.plot(epochs_range, combined_history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2.5)
    plt.axvline(x=initial_epochs_trained, color="gray", linestyle="--", alpha=0.7, label="Phase 2 Fine-Tuning Start")
    plt.title("Medicinal Flower Model - Training vs Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, combined_history["loss"], label="Training Loss", color="#1f77b4", linewidth=2.5)
    plt.plot(epochs_range, combined_history["val_loss"], label="Validation Loss", color="#d62728", linewidth=2.5)
    plt.axvline(x=initial_epochs_trained, color="gray", linestyle="--", alpha=0.7, label="Phase 2 Fine-Tuning Start")
    plt.title("Medicinal Flower Model - Training vs Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10, fontweight="bold")
    plt.ylabel("Loss", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(GRAPHS_PATH, dpi=300)
    plt.close()
    print(f"[OK] Training vs Validation graphs saved to: {GRAPHS_PATH}")

    # 12. Print Final Summary
    print("\n========================================")
    print("          FINAL TRAINING SUMMARY        ")
    print("========================================")
    print(f"Number of Training Images   : {total_train_images:,}")
    print(f"Number of Validation Images : {total_val_images:,}")
    print(f"Total Epochs Trained        : {len(epochs_range)}")
    print(f"Best Epoch (Min Val Loss)   : Epoch #{best_epoch_num}")
    print(f"Best Training Accuracy      : {best_train_acc * 100:.2f}%")
    print(f"Best Validation Accuracy    : {best_val_acc * 100:.2f}%")
    print(f"Best Validation Loss        : {best_val_loss:.4f}")
    print(f"Saved Best Model Path       : {MODEL_PATH}")
    print("========================================\n")

if __name__ == "__main__":
    main()
