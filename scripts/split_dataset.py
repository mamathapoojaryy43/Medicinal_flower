import json
import os
import random
import shutil

# Dynamic Dataset Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "combined_dataset")
DEST_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "training", "results")
JSON_STATS_PATH = os.path.join(RESULTS_DIR, "split_statistics.json")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def is_valid_image(filename):
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS

def get_unique_destination_path(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate

def main():
    print("========================================")
    print("      MEDICINAL FLOWER DATASET SPLIT    ")
    print("========================================")
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Target Directory: {DEST_DIR}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Split Ratios: {TRAIN_RATIO*100:.0f}% Train / {VAL_RATIO*100:.0f}% Val / {TEST_RATIO*100:.0f}% Test\n")

    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source dataset directory not found: {SOURCE_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Discover all class folders
    class_folders = [
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ]
    class_folders.sort()

    print(f"Found {len(class_folders)} class folders in source dataset.")
    assert len(class_folders) == 31, f"Expected 31 classes, but found {len(class_folders)}"

    # 2. Reset or create destination split directories
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(DEST_DIR, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
        for class_name in class_folders:
            os.makedirs(os.path.join(split_path, class_name), exist_ok=True)

    # Track split details
    split_details = {}
    total_train_count = 0
    total_val_count = 0
    total_test_count = 0

    # Tracking paths copied per split to verify zero overlap
    train_source_files = set()
    val_source_files = set()
    test_source_files = set()

    # 3. Process each class
    for class_name in class_folders:
        class_src_dir = os.path.join(SOURCE_DIR, class_name)
        images = [f for f in os.listdir(class_src_dir) if is_valid_image(f)]
        images.sort()  # Sort before shuffle for cross-platform determinism

        # Set random seed per class iteration for safety & reproducibility
        random.seed(RANDOM_SEED)
        random.shuffle(images)

        total_class_images = len(images)
        assert total_class_images > 0, f"Class {class_name} has 0 images!"

        train_end = int(total_class_images * TRAIN_RATIO)
        val_end = train_end + int(total_class_images * VAL_RATIO)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Ensure every class appears in all 3 splits
        assert len(train_imgs) > 0, f"Train split for {class_name} is empty!"
        assert len(val_imgs) > 0, f"Validation split for {class_name} is empty!"
        assert len(test_imgs) > 0, f"Test split for {class_name} is empty!"
        assert (len(train_imgs) + len(val_imgs) + len(test_imgs)) == total_class_images, \
            f"Image count mismatch for {class_name}!"

        # Copy train images
        train_dst_dir = os.path.join(DEST_DIR, "train", class_name)
        for img in train_imgs:
            src_path = os.path.join(class_src_dir, img)
            dst_path = get_unique_destination_path(train_dst_dir, img)
            shutil.copy2(src_path, dst_path)
            train_source_files.add(src_path)

        # Copy validation images
        val_dst_dir = os.path.join(DEST_DIR, "validation", class_name)
        for img in val_imgs:
            src_path = os.path.join(class_src_dir, img)
            dst_path = get_unique_destination_path(val_dst_dir, img)
            shutil.copy2(src_path, dst_path)
            val_source_files.add(src_path)

        # Copy test images
        test_dst_dir = os.path.join(DEST_DIR, "test", class_name)
        for img in test_imgs:
            src_path = os.path.join(class_src_dir, img)
            dst_path = get_unique_destination_path(test_dst_dir, img)
            shutil.copy2(src_path, dst_path)
            test_source_files.add(src_path)

        train_c = len(train_imgs)
        val_c = len(val_imgs)
        test_c = len(test_imgs)

        total_train_count += train_c
        total_val_count += val_c
        total_test_count += test_c

        split_details[class_name] = {
            "train": train_c,
            "validation": val_c,
            "test": test_c,
            "total": total_class_images
        }

    overall_total = total_train_count + total_val_count + total_test_count

    # 4. Print Class-wise table
    print(f"\n{'Class Name':<25} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 67)
    for class_name, counts in split_details.items():
        print(f"{class_name:<25} | {counts['train']:<8} | {counts['validation']:<8} | {counts['test']:<8} | {counts['total']:<8}")
    print("-" * 67)
    print(f"{'TOTALS':<25} | {total_train_count:<8} | {total_val_count:<8} | {total_test_count:<8} | {overall_total:<8}\n")

    # 5. Verifications
    print("========================================")
    print("           VERIFICATION CHECKS          ")
    print("========================================")

    # Check total images count
    print(f"Overall Total Images: {overall_total}")
    assert overall_total == 9516, f"Expected 9,516 total images, got {overall_total}"
    print("[OK] train + validation + test = 9,516")

    # Check class counts per split
    train_classes_count = len([c for c in class_folders if split_details[c]["train"] > 0])
    val_classes_count = len([c for c in class_folders if split_details[c]["validation"] > 0])
    test_classes_count = len([c for c in class_folders if split_details[c]["test"] > 0])

    print(f"Classes in Train Split: {train_classes_count}")
    print(f"Classes in Validation Split: {val_classes_count}")
    print(f"Classes in Test Split: {test_classes_count}")

    assert train_classes_count == 31, "Train split does not contain all 31 classes!"
    assert val_classes_count == 31, "Validation split does not contain all 31 classes!"
    assert test_classes_count == 31, "Test split does not contain all 31 classes!"
    print("[OK] Every class appears in all three splits.")

    # Check disjoint splits (no overlap)
    overlap_train_val = train_source_files.intersection(val_source_files)
    overlap_train_test = train_source_files.intersection(test_source_files)
    overlap_val_test = val_source_files.intersection(test_source_files)

    assert len(overlap_train_val) == 0, f"Found {len(overlap_train_val)} overlapping images between Train & Val!"
    assert len(overlap_train_test) == 0, f"Found {len(overlap_train_test)} overlapping images between Train & Test!"
    assert len(overlap_val_test) == 0, f"Found {len(overlap_val_test)} overlapping images between Val & Test!"
    print("[OK] No image appears in more than one split (0 overlap verified).")

    # 6. Save JSON Statistics
    stats_data = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO
        },
        "totals": {
            "train": total_train_count,
            "validation": total_val_count,
            "test": total_test_count,
            "overall": overall_total
        },
        "class_counts_per_split": {
            "train_classes": train_classes_count,
            "validation_classes": val_classes_count,
            "test_classes": test_classes_count
        },
        "class_details": split_details
    }

    with open(JSON_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=4)

    print(f"[OK] Split statistics saved to: {JSON_STATS_PATH}")
    print("========================================\n")

if __name__ == "__main__":
    main()
