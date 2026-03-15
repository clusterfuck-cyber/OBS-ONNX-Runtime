"""
Inspect an ONNX model file to check for extreme dimensions or corrupted metadata.
This helps diagnose the buffer overrun crash in obs-onnx-plugin.dll.
"""

import sys
import os

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Installing ONNX package...")
    import subprocess
    subprocess.check_call(["pip", "install", "onnx"])
    import onnx
    from onnx import numpy_helper

def inspect_model(model_path):
    """Inspect an ONNX model and report potentially problematic dimensions."""
    
    if not os.path.exists(model_path):
        print(f"✗ ERROR: Model file not found: {model_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"INSPECTING: {os.path.basename(model_path)}")
    print(f"{'='*70}")
    print(f"Path: {model_path}")
    print(f"Size: {os.path.getsize(model_path):,} bytes")
    
    try:
        # Load model
        model = onnx.load(model_path)
        print("✓ Model loaded successfully")
        
        # Check model validity
        try:
            onnx.checker.check_model(model)
            print("✓ Model passes ONNX validity checks")
        except Exception as e:
            print(f"✗ WARNING: Model validation failed: {e}")
        
        # Get graph
        graph = model.graph
        
        # Inspect inputs
        print(f"\n{'─'*70}")
        print("INPUT TENSORS:")
        print(f"{'─'*70}")
        
        for i, input_tensor in enumerate(graph.input):
            print(f"\n  Input #{i}: {input_tensor.name}")
            print(f"    Type: {onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)}")
            
            shape = []
            total_elements = 1
            has_dynamic = False
            suspicious = False
            
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    dim_val = dim.dim_value
                    shape.append(str(dim_val))
                    
                    # Check for suspicious dimensions
                    if dim_val < 0:
                        print(f"    ✗ CRITICAL: Negative dimension detected: {dim_val}")
                        suspicious = True
                    elif dim_val > 10000:
                        print(f"    ⚠ WARNING: Very large dimension: {dim_val}")
                        suspicious = True
                    elif dim_val == 0:
                        print(f"    ⚠ WARNING: Zero dimension detected")
                        suspicious = True
                    else:
                        total_elements *= dim_val
                        
                elif dim.HasField('dim_param'):
                    shape.append(f"{dim.dim_param} (dynamic)")
                    has_dynamic = True
                else:
                    shape.append("? (unknown)")
                    has_dynamic = True
            
            print(f"    Shape: [{', '.join(shape)}]")
            
            if not has_dynamic:
                size_bytes = total_elements * 4  # Assuming float32
                print(f"    Elements: {total_elements:,}")
                print(f"    Size: {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
                
                # Check for overflow risk
                if total_elements > 256 * 1024 * 1024:  # > 1GB
                    print(f"    ✗ CRITICAL: Tensor size exceeds 1GB limit!")
                    suspicious = True
                elif total_elements > 100 * 1024 * 1024:  # > 400MB
                    print(f"    ⚠ WARNING: Very large tensor (>{total_elements * 4 / (1024*1024):.0f}MB)")
            
            if has_dynamic:
                print(f"    ⚠ NOTE: Has dynamic dimensions (shape resolved at runtime)")
        
        # Inspect outputs
        print(f"\n{'─'*70}")
        print("OUTPUT TENSORS:")
        print(f"{'─'*70}")
        
        for i, output_tensor in enumerate(graph.output):
            print(f"\n  Output #{i}: {output_tensor.name}")
            print(f"    Type: {onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)}")
            
            shape = []
            total_elements = 1
            has_dynamic = False
            
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    dim_val = dim.dim_value
                    shape.append(str(dim_val))
                    
                    if dim_val < 0:
                        print(f"    ✗ CRITICAL: Negative dimension: {dim_val}")
                    elif dim_val > 10000:
                        print(f"    ⚠ WARNING: Very large dimension: {dim_val}")
                    elif dim_val > 0:
                        total_elements *= dim_val
                        
                elif dim.HasField('dim_param'):
                    shape.append(f"{dim.dim_param} (dynamic)")
                    has_dynamic = True
                else:
                    shape.append("? (unknown)")
                    has_dynamic = True
            
            print(f"    Shape: [{', '.join(shape)}]")
            
            if not has_dynamic:
                size_bytes = total_elements * 4
                print(f"    Elements: {total_elements:,}")
                print(f"    Size: {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
                
                if total_elements > 256 * 1024 * 1024:
                    print(f"    ✗ CRITICAL: Tensor size exceeds 1GB limit!")
        
        # Model info
        print(f"\n{'─'*70}")
        print("MODEL INFO:")
        print(f"{'─'*70}")
        print(f"  Producer: {model.producer_name}")
        print(f"  Opset version: {model.opset_import[0].version if model.opset_import else 'unknown'}")
        print(f"  Node count: {len(graph.node)}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Inspect specific model provided as argument
        model_path = sys.argv[1]
        inspect_model(model_path)
    else:
        # Inspect all models in the directory
        model_dir = r'E:\_DEV\OBSPlugins\ONNX files\Computer Vision'
        print(f"Scanning directory: {model_dir}")
        
        found_models = []
        
        # Search for .onnx files
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.onnx'):
                    found_models.append(os.path.join(root, file))
        
        if not found_models:
            print("No .onnx files found in directory.")
        else:
            print(f"\nFound {len(found_models)} ONNX model(s):\n")
            
            for i, model_path in enumerate(found_models, 1):
                print(f"\n{'#'*70}")
                print(f"MODEL {i} of {len(found_models)}")
                print(f"{'#'*70}")
                inspect_model(model_path)
        
        print(f"\n{'='*70}")
        print("INSPECTION COMPLETE")
        print(f"{'='*70}")
