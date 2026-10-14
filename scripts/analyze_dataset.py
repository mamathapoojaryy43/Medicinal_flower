import json
import os
import statistics
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Configuration & Dynamic Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "combined_dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "training", "results")
CHART_PATH = os.path.join(RESULTS_DIR, "class_distribution.png")
JSON_STATS_PATH = os.path.join(RESULTS_DIR, "dataset_statistics.json")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def is_valid_extension(filename):
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS

def check_single_image(args):
    class_name, filename, filepath = args
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, {
            "class": class_name,
            "filename": filename,
            "filepath": filepath,
            "error": str(e)
        }

def main():
    print("========================================")
    print("   MEDICINAL FLOWER DATASET ANALYSIS   ")
    print("========================================")
    print(f"Dataset path: {DATASET_DIR}\n")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Scan class folders
    class_folders = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
    class_folders.sort()

    total_classes = len(class_folders)
    class_counts = {}
    empty_folders = []
    image_tasks = []

    print(f"Scanning {total_classes} class folders...\n")

    total_images = 0
    for class_name in class_folders:
        class_path = os.path.join(DATASET_DIR, class_name)
        files = os.listdir(class_path)
        valid_files = [f for f in files if is_valid_extension(f)]

        if not valid_files:
            empty_folders.append(class_name)

        class_count = len(valid_files)
        class_counts[class_name] = class_count
        total_images += class_count

        for f in valid_files:
            file_path = os.path.join(class_path, f)
            image_tasks.append((class_name, f, file_path))

    # Parallel image integrity check using ThreadPoolExecutor
    print(f"Checking integrity of {total_images:,} images across {total_classes} classes...")
    corrupted_images = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(check_single_image, image_tasks)
        for is_valid, err_info in results:
            if not is_valid:
                corrupted_images.append(err_info)

    print("[OK] Image integrity check completed.\n")

    # 2. Print class table
    print(f"{'Class Name':<25} | {'Images':<10} | {'Percentage':<10}")
    print("-" * 52)

    class_percentages = {}
    for class_name, count in class_counts.items():
        pct = (count / total_images * 100) if total_images > 0 else 0
        class_percentages[class_name] = round(pct, 2)
        print(f"{class_name:<25} | {count:<10} | {pct:>8.2f}%")

    print("-" * 52)
    print(f"{'TOTAL':<25} | {total_images:<10} | {100.00:>8.2f}%\n")

    # 3. Calculate metrics
    counts_list = list(class_counts.values())

    min_count = min(counts_list)
    max_count = max(counts_list)
    min_classes = [c for c, count in class_counts.items() if count == min_count]
    max_classes = [c for c, count in class_counts.items() if count == max_count]

    mean_images = statistics.mean(counts_list)
    median_images = statistics.median(counts_list)

    # 4. Top 5 & Bottom 5 classes
    sorted_by_count = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_by_count[:5]
    bottom_5 = sorted_by_count[-5:]

    # Print calculations
    print("========================================")
    print("           DATASET METRICS              ")
    print("========================================")
    print(f"Total Classes: {total_classes}")
    print(f"Total Images: {total_images:,}")
    print(f"Minimum Class Size: {min_count} images ({', '.join(min_classes)})")
    print(f"Maximum Class Size: {max_count} images ({', '.join(max_classes)})")
    print(f"Mean Images per Class: {mean_images:.2f}")
    print(f"Median Images per Class: {median_images:.2f}")

    print("\nTop 5 Classes (Largest):")
    for name, count in top_5:
        pct = class_percentages[name]
        print(f"  - {name}: {count} images ({pct}%)")

    print("\nBottom 5 Classes (Smallest):")
    for name, count in bottom_5:
        pct = class_percentages[name]
        print(f"  - {name}: {count} images ({pct}%)")

    # Empty folder check
    print("\nEmpty Class Folders:")
    if empty_folders:
        for folder in empty_folders:
            print(f"  [WARNING] Empty folder: {folder}")
    else:
        print("  [OK] None found. All 31 class folders contain valid images.")

    # Image corruption check
    print("\nCorrupted/Unreadable Images:")
    if corrupted_images:
        print(f"  [WARNING] Found {len(corrupted_images)} corrupted image(s):")
        for item in corrupted_images:
            print(f"    - {item['filepath']}: {item['error']}")
    else:
        print(f"  [OK] None found. All {total_images:,} images verified readable using Pillow.")

    # 5. Generate Matplotlib Bar Chart
    plt.figure(figsize=(14, 8))
    sorted_classes = [item[0] for item in sorted_by_count]
    sorted_counts = [item[1] for item in sorted_by_count]

    bars = plt.bar(sorted_classes, sorted_counts, color="#2b5c8f", edgecolor="#1e3d59")
    plt.xlabel("Class Name", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Number of Images", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Combined Medicinal Flower Dataset - Class Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=75, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_count * 0.01,
            f"{height}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()
    print(f"\n[OK] Class distribution bar chart saved to: {CHART_PATH}")

    # 6. Save JSON Statistics
    stats_data = {
        "total_classes": total_classes,
        "total_images": total_images,
        "metrics": {
            "min_class_size": {
                "count": min_count,
                "classes": min_classes
            },
            "max_class_size": {
                "count": max_count,
                "classes": max_classes
            },
            "mean_images_per_class": round(mean_images, 2),
            "median_images_per_class": float(median_images)
        },
        "top_5_classes": [{"class": c, "count": count, "percentage": class_percentages[c]} for c, count in top_5],
        "bottom_5_classes": [{"class": c, "count": count, "percentage": class_percentages[c]} for c, count in bottom_5],
        "empty_folders": empty_folders,
        "corrupted_images_count": len(corrupted_images),
        "corrupted_images": corrupted_images,
        "class_details": {
            c: {
                "image_count": class_counts[c],
                "percentage": class_percentages[c]
            } for c in class_folders
        }
    }

    with open(JSON_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=4)

    print(f"[OK] Complete dataset statistics saved to: {JSON_STATS_PATH}")

    # 7. Final Summary
    print("\n========================================")
    print("           FINAL SUMMARY                ")
    print("========================================")
    print(f"Total classes analyzed: {total_classes}")
    print(f"Total valid images analyzed: {total_images:,}")
    print(f"Empty class folders: {len(empty_folders)}")
    print(f"Corrupted image files: {len(corrupted_images)}")
    print(f"Min class size: {min_count} ({min_classes[0]})")
    print(f"Max class size: {max_count} ({max_classes[0]})")
    print(f"Mean class size: {mean_images:.2f}")
    print(f"Median class size: {median_images:.2f}")
    print("========================================")

if __name__ == "__main__":
    main()
