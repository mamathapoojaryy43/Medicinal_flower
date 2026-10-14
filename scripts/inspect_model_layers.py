import os
import sys
import json
import tensorflow as tf

def inspect_model():
    print("=" * 70)
    print("      MEDICINAL FLOWER MODEL ARCHITECTURE INSPECTION       ")
    print("=" * 70)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "medicinal_flower_model.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("\n[FULL MODEL LAYERS]")
    print(f"{'Index':<6} | {'Layer Name':<30} | {'Layer Type':<25} | {'Output Shape'}")
    print("-" * 80)
    for idx, layer in enumerate(model.layers):
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = "N/A"
        print(f"{idx:<6} | {layer.name:<30} | {layer.__class__.__name__:<25} | {shape}")
        
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break
            
    if base_model is not None:
        print(f"\n[BASE MODEL LAYERS: {base_model.name}]")
        print(f"{'Index':<6} | {'Layer Name':<35} | {'Layer Type':<25} | {'Output Shape'}")
        print("-" * 85)
        for idx, layer in enumerate(base_model.layers):
            try:
                shape = str(layer.output.shape)
            except Exception:
                shape = "N/A"
            # Highlight 4D conv/activation layers with spatial dimensions
            if len(layer.output.shape) == 4:
                print(f"{idx:<6} | {layer.name:<35} | {layer.__class__.__name__:<25} | {shape} <-- CONV / SPATIAL")
            else:
                print(f"{idx:<6} | {layer.name:<35} | {layer.__class__.__name__:<25} | {shape}")
                
    print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect_model()
