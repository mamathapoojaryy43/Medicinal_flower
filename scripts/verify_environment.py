import sys

def verify():
    print("========================================", flush=True)
    print("   ENVIRONMENT VERIFICATION REPORT      ", flush=True)
    print("========================================", flush=True)

    # 1. Python version
    py_ver = sys.version.replace("\n", " ")
    print(f"Python Executable : {sys.executable}", flush=True)
    print(f"Python Version    : {py_ver}\n", flush=True)

    # 2. Imports verification
    modules_to_test = [
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow (PIL)"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-Learn"),
        ("flask", "Flask")
    ]

    print("Verifying Module Imports...", flush=True)
    print("-" * 50, flush=True)
    failed = False
    import importlib.metadata
    for mod_name, label in modules_to_test:
        try:
            mod = __import__(mod_name)
            try:
                ver = importlib.metadata.version(mod_name)
            except Exception:
                ver = getattr(mod, "__version__", "Available")
            print(f"  [OK] {label:<16}: {ver}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {label:<14}: IMPORT ERROR -> {e}", flush=True)
            failed = True

    print("-" * 50, flush=True)
    if failed:
        print("\n[ERROR] One or more required imports failed!", flush=True)
        sys.exit(1)

    import tensorflow as tf
    import keras

    # 3. TensorFlow & Keras specs
    print(f"\nTensorFlow Version : {tf.__version__}", flush=True)
    print(f"Keras Version      : {keras.__version__}", flush=True)

    # 4. GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Availability   : AVAILABLE ({len(gpus)} GPU(s) detected)", flush=True)
        for i, gpu in enumerate(gpus):
            print(f"  - GPU #{i+1}: {gpu.name}", flush=True)
    else:
        print("GPU Availability   : NOT AVAILABLE (CPU execution only)", flush=True)

    print("\n========================================", flush=True)
    print(" [OK] All required imports verified successfully!", flush=True)
    print("========================================\n", flush=True)

if __name__ == "__main__":
    verify()
