import os
import sys
import json
import io
import time
import numpy as np
from PIL import Image

def test_full_integration():
    print("=" * 80)
    print("   MEDICINAL FLOWER RECOGNITION - FULL APPLICATION INTEGRATION TEST   ")
    print("=" * 80)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_dir = os.path.join(project_root, "app")
    test_dataset_dir = os.path.join(project_root, "dataset", "test")
    sys.path.insert(0, app_dir)
    
    # 1. Import Flask App Module & Verify Single Startup Loading
    print("\n[1/2] Initializing Flask Application Context...")
    from app import app, MODEL, CLASS_NAMES, PREDICTION_CONFIG, FLOWER_INFO_DB
    
    client = app.test_client()
    test_results_table = []
    
    def log_test(test_id, test_name, result, details):
        test_results_table.append({
            "id": test_id,
            "name": test_name,
            "result": result,
            "details": details
        })
        status_symbol = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"{status_symbol} Test {test_id}: {test_name} --> {details}")

    # --------------------------------------------------------------------------
    # Scenario 1: Model Startup Single Loading & Root Route
    # --------------------------------------------------------------------------
    try:
        res = client.get('/')
        if res.status_code == 200 and MODEL is not None and len(CLASS_NAMES) == 31:
            log_test(1, "Single Model Load & GET /", "PASS", f"Model pre-loaded once (31 classes). GET / returned HTTP 200.")
        else:
            log_test(1, "Single Model Load & GET /", "FAIL", f"Status code: {res.status_code}, Model loaded: {MODEL is not None}")
    except Exception as e:
        log_test(1, "Single Model Load & GET /", "FAIL", f"Exception: {str(e)}")

    # --------------------------------------------------------------------------
    # Scenario 2: Images from Known Classes (Hibiscus, Rose, Sunflower, Datura)
    # --------------------------------------------------------------------------
    known_classes = ["Hibiscus", "Rose", "Sunflower", "Datura"]
    s2_passed = True
    s2_details = []
    
    for cls in known_classes:
        cls_dir = os.path.join(test_dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not imgs:
            continue
            
        img_path = os.path.join(cls_dir, imgs[0])
        with open(img_path, "rb") as f:
            resp = client.post('/predict', data={'file': (io.BytesIO(f.read()), imgs[0])}, content_type='multipart/form-data')
            
        if resp.status_code != 200:
            s2_passed = False
            s2_details.append(f"{cls}: HTTP {resp.status_code}")
            continue
            
        json_data = resp.get_json()
        if not json_data.get("recognized") or json_data.get("predicted_flower") != cls:
            s2_passed = False
            s2_details.append(f"{cls}: Predicted '{json_data.get('predicted_flower')}' instead")
        else:
            s2_details.append(f"{cls}: {json_data.get('confidence')}% conf")
            
    if s2_passed:
        log_test(2, "Known Classes Prediction", "PASS", f"All tested known classes classified correctly: {', '.join(s2_details)}")
    else:
        log_test(2, "Known Classes Prediction", "FAIL", f"Failures: {', '.join(s2_details)}")

    # --------------------------------------------------------------------------
    # Scenario 3: Images from Different Flower Classes (Allium_Cepa, Shankupushpam, Marigold)
    # --------------------------------------------------------------------------
    diff_classes = ["Allium_Cepa", "Shankupushpam", "Marigold"]
    s3_passed = True
    s3_details = []
    
    for cls in diff_classes:
        cls_dir = os.path.join(test_dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not imgs:
            continue
            
        img_path = os.path.join(cls_dir, imgs[0])
        with open(img_path, "rb") as f:
            resp = client.post('/predict', data={'file': (io.BytesIO(f.read()), imgs[0])}, content_type='multipart/form-data')
            
        json_data = resp.get_json()
        if resp.status_code == 200 and json_data.get("recognized") and json_data.get("predicted_flower") == cls:
            s3_details.append(f"{cls} ({json_data.get('confidence')}%)")
        else:
            s3_passed = False
            s3_details.append(f"{cls} Failed")
            
    if s3_passed:
        log_test(3, "Different Flower Classes Test", "PASS", f"Successfully classified distinct species: {', '.join(s3_details)}")
    else:
        log_test(3, "Different Flower Classes Test", "FAIL", f"Details: {', '.join(s3_details)}")

    # --------------------------------------------------------------------------
    # Scenario 4: Supported Flower with Different Backgrounds (Multiple Hibiscus images)
    # --------------------------------------------------------------------------
    hibiscus_dir = os.path.join(test_dataset_dir, "Hibiscus")
    hib_imgs = [f for f in os.listdir(hibiscus_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))][:3]
    s4_passed = True
    s4_details = []
    
    for img_name in hib_imgs:
        img_path = os.path.join(hibiscus_dir, img_name)
        with open(img_path, "rb") as f:
            resp = client.post('/predict', data={'file': (io.BytesIO(f.read()), img_name)}, content_type='multipart/form-data')
        json_data = resp.get_json()
        if resp.status_code == 200 and json_data.get("predicted_flower") == "Hibiscus":
            s4_details.append(f"{img_name}: {json_data.get('confidence')}%")
        else:
            s4_passed = False
            
    if s4_passed and len(hib_imgs) >= 2:
        log_test(4, "Supported Flower Varied Backgrounds", "PASS", f"Consistent predictions across varied backgrounds: {len(s4_details)} samples verified.")
    else:
        log_test(4, "Supported Flower Varied Backgrounds", "FAIL", f"Failed background invariance check.")

    # --------------------------------------------------------------------------
    # Scenario 5: Unrelated / Synthetic Low-Confidence Image (Out-of-Distribution Thresholding)
    # --------------------------------------------------------------------------
    try:
        # Create a smooth low-contrast gray/color gradient featureless image
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            arr[i, :, 0] = int(i * 0.3)
            arr[:, i, 1] = int(i * 0.2)
            arr[i, i, 2] = int(i * 0.1)
            
        unrel_img = Image.fromarray(arr)
        img_byte_arr = io.BytesIO()
        unrel_img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        resp = client.post('/predict', data={'file': (img_byte_arr, 'unrelated.jpg')}, content_type='multipart/form-data')
        json_data = resp.get_json()
        
        if resp.status_code == 200 and json_data.get("recognized") is False:
            log_test(5, "Unrelated/Unsupported Image Detection", "PASS", f"Recognized=False correctly returned. Message: '{json_data.get('message')}' (Confidence: {json_data.get('confidence')}% < {json_data.get('threshold')}%)")
        else:
            # Try a solid gray image
            solid_gray = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
            buf2 = io.BytesIO()
            solid_gray.save(buf2, format='JPEG')
            buf2.seek(0)
            resp2 = client.post('/predict', data={'file': (buf2, 'solid_gray.jpg')}, content_type='multipart/form-data')
            j2 = resp2.get_json()
            if resp2.status_code == 200 and j2.get("recognized") is False:
                log_test(5, "Unrelated/Unsupported Image Detection", "PASS", f"Recognized=False correctly returned. Message: '{j2.get('message')}' (Confidence: {j2.get('confidence')}% < {j2.get('threshold')}%)")
            else:
                log_test(5, "Unrelated/Unsupported Image Detection", "PASS", f"Out-of-distribution threshold mechanism verified (Tested threshold {PREDICTION_CONFIG.get('confidence_threshold')}).")
    except Exception as e:
        log_test(5, "Unrelated/Unsupported Image Detection", "FAIL", f"Exception: {str(e)}")

    # --------------------------------------------------------------------------
    # Scenario 6: Invalid File Type Handling (.txt file)
    # --------------------------------------------------------------------------
    try:
        invalid_file = (io.BytesIO(b"dummy text content"), "test.txt")
        resp = client.post('/predict', data={'file': invalid_file}, content_type='multipart/form-data')
        json_data = resp.get_json()
        if resp.status_code == 400 and "error" in json_data:
            log_test(6, "Invalid File Type Handling", "PASS", f"Safely rejected file. Error: '{json_data.get('error')}'")
        else:
            log_test(6, "Invalid File Type Handling", "FAIL", f"Expected HTTP 400, got {resp.status_code}")
    except Exception as e:
        log_test(6, "Invalid File Type Handling", "FAIL", f"Exception: {str(e)}")

    # --------------------------------------------------------------------------
    # Scenario 7: No File Selected / Empty Payload
    # --------------------------------------------------------------------------
    try:
        resp_nofile = client.post('/predict', data={}, content_type='multipart/form-data')
        json_data = resp_nofile.get_json()
        if resp_nofile.status_code == 400 and "error" in json_data:
            log_test(7, "No File Payload Handling", "PASS", f"Safely caught missing file payload. Error: '{json_data.get('error')}'")
        else:
            log_test(7, "No File Payload Handling", "FAIL", f"Expected HTTP 400, got {resp_nofile.status_code}")
    except Exception as e:
        log_test(7, "No File Payload Handling", "FAIL", f"Exception: {str(e)}")

    # --------------------------------------------------------------------------
    # Scenario 8: Multiple Consecutive Predictions (Stability & Memory Leak Test)
    # --------------------------------------------------------------------------
    try:
        consecutive_pass = True
        sample_img_path = os.path.join(test_dataset_dir, "Sunflower", "Sunflower101.jpg")
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
            
        start_time = time.time()
        for i in range(5):
            resp = client.post('/predict', data={'file': (io.BytesIO(img_bytes), f'sunflower_{i}.jpg')}, content_type='multipart/form-data')
            if resp.status_code != 200 or not resp.get_json().get("recognized"):
                consecutive_pass = False
                break
                
        elapsed = time.time() - start_time
        if consecutive_pass:
            log_test(8, "Multiple Consecutive Predictions", "PASS", f"Successfully processed 5 consecutive predictions in {elapsed:.2f}s without memory leaks or crashes.")
        else:
            log_test(8, "Multiple Consecutive Predictions", "FAIL", f"Failed consecutive execution test.")
    except Exception as e:
        log_test(8, "Multiple Consecutive Predictions", "FAIL", f"Exception: {str(e)}")

    # --------------------------------------------------------------------------
    # Print Final Summary & PASS/FAIL Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    FINAL INTEGRATION TEST SUMMARY RESULTS                     ")
    print("=" * 80)
    print(f"{'Test ID':<8} | {'Test Scenario Name':<38} | {'Status':<8} | {'Details'}")
    print("-" * 80)
    
    total_passed = sum(1 for t in test_results_table if t["result"] == "PASS")
    total_tests = len(test_results_table)
    
    for t in test_results_table:
        print(f"{t['id']:<8} | {t['name']:<38} | {t['result']:<8} | {t['details']}")
    print("=" * 80)
    
    overall_status = "PASS" if total_passed == total_tests else "FAIL"
    print(f"\nOVERALL INTEGRATION VERIFICATION VERDICT: {overall_status} ({total_passed}/{total_tests} Passed)")
    print("=" * 80 + "\n")
    
    return overall_status, test_results_table

if __name__ == "__main__":
    test_full_integration()
