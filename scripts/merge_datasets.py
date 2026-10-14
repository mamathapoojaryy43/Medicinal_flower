import json
import os
import shutil

# Dynamic Dataset Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMFI_SOURCE = os.path.join(PROJECT_ROOT, "IMFI Indian Medicinal Flower Image Dataset", "IMFI Indian Medicinal Flower Image Dataset", "IMFI Dataset")
KAGGLE_SOURCE = os.path.join(PROJECT_ROOT, "dataset")
COMBINED_DEST = os.path.join(PROJECT_ROOT, "combined_dataset")
JSON_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "combined_dataset_summary.json")
JSON_SUMMARY_INSIDE = os.path.join(COMBINED_DEST, "dataset_summary.json")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

IMFI_CLASSES = [
    "Adathoda",
    "Banana",
    "Bush Clock Vine",
    "Champaka",
    "Chitrak",
    "Common Lanthana",
    "Crown Flower",
    "Datura",
    "Four O' Clock Flower",
    "Hibiscus",
    "Honey Suckle",
    "Indian Mallow",
    "Jatropha",
    "Malabar Melastome",
    "Marigold",
    "Nagapoovu",
    "Nityakalyani",
    "Pinwheel",
    "Rose",
    "Shankupushpam",
    "Spider Lily",
    "Sunflower",
    "Thechi",
    "Thumba",
    "Touch Me Not",
    "Tridax procumbens",
    "Wild potato vine",
    "Yellow Daisy"
]

KAGGLE_CLASSES = [
    "Allium_Cepa",
    "Black_cumin",
    "Sweet_pea"
]

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS

def get_unique_destination_path(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate

def count_source_images(dir_path):
    count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if is_image_file(f):
                count += 1
    return count

def verify_source_folders():
    print("Verifying source folders before merging...")
    missing_imfi = [c for c in IMFI_CLASSES if not os.path.isdir(os.path.join(IMFI_SOURCE, c))]
    if missing_imfi:
        raise FileNotFoundError(f"Missing IMFI folders: {missing_imfi}")
    
    found_kaggle = set()
    for root, dirs, _ in os.walk(KAGGLE_SOURCE):
        for d in dirs:
            if d in KAGGLE_CLASSES:
                found_kaggle.add(d)
    
    missing_kaggle = set(KAGGLE_CLASSES) - found_kaggle
    if missing_kaggle:
        raise FileNotFoundError(f"Missing Kaggle folders: {missing_kaggle}")
    
    print("[OK] All 28 IMFI class folders verified.")
    print("[OK] All 3 Kaggle class folders verified.")

def main():
    # 1. Pre-copy verification
    verify_source_folders()
    
    imfi_initial_count = count_source_images(IMFI_SOURCE)
    kaggle_initial_count = count_source_images(KAGGLE_SOURCE)
    print(f"Initial IMFI source image count: {imfi_initial_count}")
    print(f"Initial Kaggle source image count: {kaggle_initial_count}")
    
    if os.path.exists(COMBINED_DEST):
        print(f"Clearing existing combined_dataset folder at: {COMBINED_DEST}")
        shutil.rmtree(COMBINED_DEST)
    os.makedirs(COMBINED_DEST, exist_ok=True)
    
    class_counts = {}
    
    # 2. Copy IMFI images
    print("\n--- Copying IMFI Dataset Images ---")
    for class_name in IMFI_CLASSES:
        src_dir = os.path.join(IMFI_SOURCE, class_name)
        dst_dir = os.path.join(COMBINED_DEST, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        copied_count = 0
        for f in os.listdir(src_dir):
            if is_image_file(f):
                src_path = os.path.join(src_dir, f)
                dst_path = get_unique_destination_path(dst_dir, f)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        
        class_counts[class_name] = copied_count
        print(f"{class_name}: {copied_count} images")
        
    # 3. Copy Kaggle images
    print("\n--- Copying Kaggle Dataset Images ---")
    for class_name in KAGGLE_CLASSES:
        dst_dir = os.path.join(COMBINED_DEST, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        copied_count = 0
        for root, _, files in os.walk(KAGGLE_SOURCE):
            if os.path.basename(root) == class_name:
                for f in files:
                    if is_image_file(f):
                        src_path = os.path.join(root, f)
                        dst_path = get_unique_destination_path(dst_dir, f)
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                        
        class_counts[class_name] = copied_count
        print(f"{class_name}: {copied_count} images")
        
    # 4. Post-merge combined folder verification
    combined_folders = [
        d for d in os.listdir(COMBINED_DEST)
        if os.path.isdir(os.path.join(COMBINED_DEST, d))
    ]
    total_classes = len(combined_folders)
    total_images = sum(class_counts.values())
    
    print("\n--- Combined Dataset Verification ---")
    print(f"Total class folders found in combined dataset: {total_classes}")
    assert total_classes == 31, f"Expected 31 class folders, but found {total_classes}"
    print("[OK] Exactly 31 class folders exist in combined dataset.")
    
    # 5. Create JSON summary file
    summary_data = {
        "imfi_classes_count": len(IMFI_CLASSES),
        "kaggle_classes_count": len(KAGGLE_CLASSES),
        "total_classes": total_classes,
        "total_images": total_images,
        "class_image_counts": class_counts
    }
    
    with open(JSON_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        
    with open(JSON_SUMMARY_INSIDE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"[OK] JSON summary written to: {JSON_SUMMARY_PATH}")
    print(f"[OK] JSON summary written to: {JSON_SUMMARY_INSIDE}")
    
    # 6. Verify source datasets are unchanged
    imfi_final_count = count_source_images(IMFI_SOURCE)
    kaggle_final_count = count_source_images(KAGGLE_SOURCE)
    
    assert imfi_initial_count == imfi_final_count, "IMFI source dataset was modified!"
    assert kaggle_initial_count == kaggle_final_count, "Kaggle source dataset was modified!"
    print("[OK] Source datasets verified unchanged.")
    
    # 7. Print final summary
    print("\nIMFI classes: 28")
    print("Kaggle classes: 3")
    print(f"Combined classes: {total_classes}")
    print(f"Total images: {total_images:,}")

if __name__ == "__main__":
    main()
