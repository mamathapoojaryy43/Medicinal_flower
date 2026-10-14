import os
import sys
import json

def validate_info():
    print("=" * 65)
    print("      FLOWER INFORMATION DATABASE VALIDATION REPORT        ")
    print("=" * 65)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    class_mapping_path = os.path.join(project_root, "models", "class_names.json")
    info_db_path = os.path.join(project_root, "app", "flower_information.json")
    
    # 1. Check file existence
    if not os.path.exists(class_mapping_path):
        print(f"[FAIL] Master class mapping not found at: {class_mapping_path}")
        sys.exit(1)
    if not os.path.exists(info_db_path):
        print(f"[FAIL] Flower information DB not found at: {info_db_path}")
        sys.exit(1)
        
    # 2. Load JSON files
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        master_data = json.load(f)
    master_classes = set(master_data["class_names"])
    num_master_classes = len(master_classes)
    
    with open(info_db_path, "r", encoding="utf-8") as f:
        info_data = json.load(f)
    db_keys = set(info_data.keys())
    num_db_keys = len(db_keys)
    
    print(f"Master ML Classes Count : {num_master_classes}")
    print(f"Information DB Keys     : {num_db_keys}")
    print("-" * 65)
    
    # 3. Check for missing or extra classes
    missing_classes = master_classes - db_keys
    extra_classes = db_keys - master_classes
    
    errors = []
    if missing_classes:
        errors.append(f"Missing entries for {len(missing_classes)} ML classes: {sorted(list(missing_classes))}")
    if extra_classes:
        errors.append(f"Found {len(extra_classes)} extra/unrecognized keys: {sorted(list(extra_classes))}")
        
    # 4. Check required fields for each entry
    required_fields = ["class_name", "common_name", "scientific_name", "short_botanical_description", "traditional_uses"]
    invalid_field_entries = []
    
    for cls in master_classes.intersection(db_keys):
        entry = info_data[cls]
        for field in required_fields:
            if field not in entry or not str(entry[field]).strip():
                invalid_field_entries.append(f"Class '{cls}' missing or empty required field '{field}'")
                
    if invalid_field_entries:
        errors.extend(invalid_field_entries)
        
    # 5. Print Detailed Status Table
    print(f"{'#':<3} | {'Dataset Class Name':<25} | {'Scientific Name':<26} | {'Status':<8}")
    print("-" * 65)
    
    for idx, cls in enumerate(master_data["class_names"], start=1):
        if cls in info_data:
            sci_name = info_data[cls].get("scientific_name", "N/A")
            status = "OK"
        else:
            sci_name = "MISSING"
            status = "MISSING"
        print(f"{idx:<3} | {cls:<25} | {sci_name:<26} | {status:<8}")
        
    print("=" * 65)
    
    # 6. Report Verdict
    if errors:
        print("\n[FAIL] VALIDATION FAILED WITH ERRORS:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] FLOWER INFORMATION DATABASE IS FULLY VALIDATED!")
        print("  * All 31 ML classes exist in app/flower_information.json")
        print("  * No missing classes")
        print("  * No extra/unrecognized classes")
        print("  * All required botanical fields are populated")
        print("=" * 65 + "\n")

if __name__ == "__main__":
    validate_info()
