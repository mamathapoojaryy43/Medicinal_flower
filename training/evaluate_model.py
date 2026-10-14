import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

def evaluate_model():
    print("=" * 60)
    print("      MEDICINAL FLOWER RECOGNITION - MODEL EVALUATION       ")
    print("=" * 60)
    
    # 1. Define dynamic paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_dir = os.path.join(project_root, "dataset", "test")
    model_path = os.path.join(project_root, "models", "medicinal_flower_model.keras")
    class_mapping_path = os.path.join(project_root, "models", "class_names.json")
    results_dir = os.path.join(project_root, "training", "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Check existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found at: {class_mapping_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test dataset directory not found at: {test_dir}")
        
    # 2. Load Class Mapping
    print(f"\n[1/5] Loading class mapping from: {class_mapping_path}")
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    
    class_names = class_info["class_names"]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} class names.")
    
    # 3. Load Trained Model
    print(f"\n[2/5] Loading saved Keras model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    # 4. Prepare Test Dataset
    print(f"\n[3/5] Loading test images from: {test_dir}")
    batch_size = 32
    img_size = (224, 224)
    
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,  # Ensures exact class index matching
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False  # Do NOT shuffle test data for evaluation
    )
    
    # Note: Preprocessing (mobilenet_v2.preprocess_input) is built directly into the model architecture
    
    # Collect true labels
    y_true_list = []
    for _, labels in test_dataset:
        y_true_list.append(labels.numpy())
    
    y_true_onehot = np.vstack(y_true_list)
    y_true = np.argmax(y_true_onehot, axis=1)
    num_test_images = len(y_true)
    
    print(f"Total Test Images: {num_test_images} across {num_classes} classes.")
    
    # 5. Evaluate Model (Test Loss & Accuracy)
    print("\n[4/5] Running model evaluation on test set...")
    eval_results = model.evaluate(test_dataset, verbose=1)
    test_loss = float(eval_results[0])
    test_accuracy = float(eval_results[1])
    
    # Generate predictions
    print("\nGenerating model predictions...")
    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 6. Calculate Detailed Metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    class_performance_list = []
    for i, cls_name in enumerate(class_names):
        metrics_dict = {
            "precision": float(p_per_class[i]),
            "recall": float(r_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(support_per_class[i])
        }
        per_class_metrics[cls_name] = metrics_dict
        class_performance_list.append((cls_name, float(f1_per_class[i]), float(p_per_class[i]), float(r_per_class[i]), int(support_per_class[i])))
        
    # Sort classes by F1-score
    class_performance_sorted = sorted(class_performance_list, key=lambda x: x[1], reverse=True)
    top_5_classes = class_performance_sorted[:5]
    bottom_5_classes = class_performance_sorted[-5:]
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Most confused class pairs (off-diagonal > 0)
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j])
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)
    
    # 7. Save Confusion Matrix Plot
    print("\n[5/5] Exporting evaluation artifacts...")
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    
    fig, ax = plt.subplots(figsize=(20, 18), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix - Medicinal Flower Recognition (Test Set)",
        ylabel="True Class",
        xlabel="Predicted Class"
    )
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    # Annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, format(val, 'd'),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8,
                    fontweight="bold" if i == j else "normal"
                )
                
    fig.tight_layout()
    plt.savefig(cm_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f" - Saved Confusion Matrix plot to: {cm_plot_path}")
    
    # 8. Save Classification Report TXT
    clf_report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    txt_report_path = os.path.join(results_dir, "classification_report.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("MEDICINAL FLOWER RECOGNITION - TEST SET CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(clf_report_str)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("TOP 5 CLASSES WITH HIGHEST F1-SCORE:\n")
        for cls, f1, p, r, sup in top_5_classes:
            f.write(f" - {cls:<25}: F1={f1:.4f}, Precision={p:.4f}, Recall={r:.4f}, Support={sup}\n")
        f.write("\nBOTTOM 5 CLASSES WITH LOWEST F1-SCORE:\n")
        for cls, f1, p, r, sup in bottom_5_classes:
            f.write(f" - {cls:<25}: F1={f1:.4f}, Precision={p:.4f}, Recall={r:.4f}, Support={sup}\n")
        f.write("\nMOST FREQUENTLY CONFUSED CLASS PAIRS:\n")
        if confused_pairs:
            for item in confused_pairs[:10]:
                f.write(f" - True: {item['true_class']:<22} --> Predicted: {item['predicted_class']:<22} ({item['count']} instances)\n")
        else:
            f.write(" - None! Zero classification errors on the test set.\n")
            
    print(f" - Saved Classification Report text to: {txt_report_path}")
    
    # 9. Save JSON Evaluation Results
    json_results_path = os.path.join(results_dir, "test_evaluation.json")
    evaluation_json_data = {
        "evaluation_dataset": "dataset/test",
        "num_test_images": num_test_images,
        "num_test_classes": num_classes,
        "overall_metrics": {
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1_score": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1_score": float(weighted_f1)
        },
        "top_5_highest_f1_classes": [
            {"class_name": cls, "f1_score": f1, "precision": p, "recall": r, "support": sup}
            for cls, f1, p, r, sup in top_5_classes
        ],
        "bottom_5_lowest_f1_classes": [
            {"class_name": cls, "f1_score": f1, "precision": p, "recall": r, "support": sup}
            for cls, f1, p, r, sup in bottom_5_classes
        ],
        "most_confused_pairs": confused_pairs,
        "per_class_metrics": per_class_metrics
    }
    
    with open(json_results_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_json_data, f, indent=4)
    print(f" - Saved Evaluation JSON data to: {json_results_path}")
    
    # 10. Print Final Test Results Summary
    print("\n" + "=" * 60)
    print("                  FINAL TEST RESULTS                    ")
    print("--------------------------------------------------------")
    print(f"Test images        : {num_test_images}")
    print(f"Test classes       : {num_classes}")
    print(f"Test accuracy      : {test_accuracy * 100:.2f}% ({test_accuracy:.4f})")
    print(f"Test loss          : {test_loss:.4f}")
    print(f"Macro precision    : {macro_p:.4f}")
    print(f"Macro recall       : {macro_r:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Weighted precision : {weighted_p:.4f}")
    print(f"Weighted recall    : {weighted_r:.4f}")
    print(f"Weighted F1        : {weighted_f1:.4f}")
    print("=" * 60)
    
    print("\nTop 5 Classes (Highest F1-Score):")
    for cls, f1, p, r, sup in top_5_classes:
        print(f"  * {cls:<24}: F1={f1:.4f}")
        
    print("\nBottom 5 Classes (Lowest F1-Score):")
    for cls, f1, p, r, sup in bottom_5_classes:
        print(f"  * {cls:<24}: F1={f1:.4f}")
        
    print("\nMost Confused Class Pairs:")
    if confused_pairs:
        for item in confused_pairs[:5]:
            print(f"  * True: {item['true_class']:<20} -> Pred: {item['predicted_class']:<20} ({item['count']} images)")
    else:
        print("  * None! Perfect 100% classification on test set.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate_model()
