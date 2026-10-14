import os
import sys
import json
import io

def test_backend():
    print("=" * 70)
    print("      MEDICINAL FLOWER RECOGNITION - FLASK BACKEND VERIFICATION     ")
    print("=" * 70)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_dir = os.path.join(project_root, "app")
    sys.path.insert(0, app_dir)
    
    # 1. Import app module (triggers resource loading)
    print("\n[1/4] Importing Flask application module...")
    from app import app, MODEL, CLASS_NAMES, PREDICTION_CONFIG, FLOWER_INFO_DB
    
    if MODEL is None:
        print("[FAIL] Model failed to load into memory.")
        sys.exit(1)
        
    print(f"[OK] Model loaded successfully. Output classes: {len(CLASS_NAMES)}")
    print(f"[OK] Prediction config loaded. Threshold: {PREDICTION_CONFIG.get('confidence_threshold')}")
    print(f"[OK] Flower Info DB loaded. Total entries: {len(FLOWER_INFO_DB)}")
    
    # Create Flask test client
    client = app.test_client()
    
    # 2. Test GET /
    print("\n[2/4] Testing GET / endpoint...")
    response_get = client.get('/')
    assert response_get.status_code == 200, f"Expected status 200, got {response_get.status_code}"
    assert b"Medicinal Flower Recognition" in response_get.data, "Expected title in index.html response"
    print(f" -> Response Status Code : {response_get.status_code}")
    print(f" -> Rendered HTML Title  : Medicinal Flower Recognition")
    print("[OK] GET / test passed successfully.")
    
    # 3. Test POST /predict with a valid sample image
    print("\n[3/4] Testing POST /predict with valid Hibiscus test image...")
    sample_img_path = os.path.join(project_root, "dataset", "test", "Hibiscus", "Hibiscus104.jpg")
    if not os.path.exists(sample_img_path):
        print(f"[FAIL] Test image not found at: {sample_img_path}")
        sys.exit(1)
        
    with open(sample_img_path, "rb") as img_file:
        data = {
            'file': (io.BytesIO(img_file.read()), 'Hibiscus104.jpg')
        }
        response_post = client.post('/predict', data=data, content_type='multipart/form-data')
        
    assert response_post.status_code == 200, f"Expected status 200, got {response_post.status_code}"
    post_json = response_post.get_json()
    print(f" -> Response Status Code : {response_post.status_code}")
    print(f" -> Recognized           : {post_json.get('recognized')}")
    print(f" -> Predicted Flower     : {post_json.get('predicted_flower')}")
    print(f" -> Confidence           : {post_json.get('confidence')}%")
    print(f" -> Scientific Name      : {post_json.get('scientific_name')}")
    print(f" -> Uploaded Image URL   : {post_json.get('uploaded_image_url')}")
    print(f" -> Grad-CAM Image URL   : {post_json.get('gradcam_image_url')}")
    print(f" -> Top Predictions      : {post_json.get('top_predictions')[:2]}")
    
    assert post_json.get("recognized") is True, "Expected recognized = True"
    assert post_json.get("predicted_flower") == "Hibiscus", "Expected prediction 'Hibiscus'"
    assert "gradcam_image_url" in post_json, "Missing Grad-CAM image URL in response"
    
    # Verify Grad-CAM image file exists on disk
    gradcam_rel_path = post_json["gradcam_image_url"].lstrip('/')
    gradcam_full_path = os.path.join(app_dir, gradcam_rel_path)
    assert os.path.exists(gradcam_full_path), f"Grad-CAM file not found on disk at: {gradcam_full_path}"
    print(f"[OK] Verified Grad-CAM file saved at: {gradcam_full_path}")
    
    # 4. Test Invalid File Upload Handling
    print("\n[4/4] Testing POST /predict with unsupported file format...")
    invalid_data = {
        'file': (io.BytesIO(b"fake txt content"), 'document.txt')
    }
    response_invalid = client.post('/predict', data=invalid_data, content_type='multipart/form-data')
    assert response_invalid.status_code == 400, f"Expected status 400, got {response_invalid.status_code}"
    print(f" -> Response Status Code : {response_invalid.status_code}")
    print(f" -> Error Message        : {response_invalid.get_json().get('error')}")
    print("[OK] Invalid file handling test passed successfully.")
    
    print("\n" + "=" * 70)
    print("      ALL FLASK BACKEND & FRONTEND TESTS PASSED SUCCESSFULLY!       ")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_backend()
