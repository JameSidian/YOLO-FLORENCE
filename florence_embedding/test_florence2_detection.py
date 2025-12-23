#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Florence-2 Object Detection

Tests how well Florence-2 performs object detection on an image.
This is useful to verify the model works correctly before using it for embeddings.

Usage:
    python test_florence2_detection.py [--image-path path/to/image.png]
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import sys

# Workaround: Patch transformers to skip flash_attn import check
# Florence-2 models check for flash_attn during import, but have CPU fallback
# Based on: https://huggingface.co/microsoft/Florence-2-base/discussions/4
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers import dynamic_module_utils
    TRANSFORMERS_AVAILABLE = True
    
    # Patch get_imports to remove flash_attn from required imports
    original_get_imports = dynamic_module_utils.get_imports
    
    def patched_get_imports(filename):
        """Patch get_imports to remove flash_attn from imports"""
        imports = original_get_imports(filename)
        if isinstance(imports, list):
            imports = [imp for imp in imports if imp != "flash_attn" and imp != "flash-attn"]
        elif isinstance(imports, set):
            imports = imports - {"flash_attn", "flash-attn"}
        return imports
    
    dynamic_module_utils.get_imports = patched_get_imports
    
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers not installed. Install with: pip install transformers")
    sys.exit(1)

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("WARNING: supervision not installed. Install with: pip install supervision")
    print("Visualization will be limited without supervision library.")


def test_object_detection(image_path: str, model_id: str = "microsoft/Florence-2-base-ft", 
                          output_path: str = None, cache_dir: str = None):
    """Test Florence-2 object detection on an image"""
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return False
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"\nLoading Florence-2 model: {model_id}")
    print("This may take a moment on first run (downloading model)...")
    
    try:
        # Load model (Florence-2-large doesn't need revision)
        # Only base-ft needs revision='refs/pr/6'
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa"  # Force standard attention instead of flash_attn
        }
        processor_kwargs = {
            "trust_remote_code": True
        }
        
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
            processor_kwargs["cache_dir"] = cache_dir
        
        if "base-ft" in model_id.lower():
            # base-ft model requires revision='refs/pr/6'
            model_kwargs["revision"] = 'refs/pr/6'
            processor_kwargs["revision"] = 'refs/pr/6'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_id,
            **processor_kwargs
        )
        
        print("[OK] Model loaded successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load image
    print(f"\nLoading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"  Image size: {image.size}")
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        return False
    
    # Perform object detection
    print("\nRunning object detection...")
    try:
        task = "<OD>"
        text = "<OD>"
        
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(device)
        
        print("  Generating detections...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        print("  Post-processing results...")
        response = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.size
        )
        
        print("[OK] Detection complete")
        
        # Display results
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        
        if isinstance(response, dict):
            # Parse the response
            if 'OD' in response:
                od_results = response['OD']
                print(f"\nFound {len(od_results)} objects:")
                for i, obj in enumerate(od_results, 1):
                    if isinstance(obj, dict):
                        label = obj.get('label', 'unknown')
                        bbox = obj.get('bbox', [])
                        score = obj.get('score', 0.0)
                        print(f"  {i}. {label}")
                        if bbox:
                            print(f"     Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                        if score:
                            print(f"     Score: {score:.3f}")
                    else:
                        print(f"  {i}. {obj}")
                print(f"\nRaw response keys: {list(response.keys())}")
            else:
                print(f"\nResponse type: {type(response)}")
                print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
                print(f"Response preview: {str(response)[:500]}")
        else:
            print(f"\nResponse type: {type(response)}")
            print(f"Response: {response}")
        
        print("="*60)
        
        # Create visualization
        print("\nCreating visualization...")
        try:
            # Manual visualization using PIL (works regardless of supervision version)
            annotated_image = image.copy()
            from PIL import ImageDraw
            
            draw = ImageDraw.Draw(annotated_image)
            
            # Parse response manually
            if isinstance(response, dict) and '<OD>' in response:
                od_data = response['<OD>']
                bboxes = od_data.get('bboxes', [])
                labels = od_data.get('labels', [])
                
                print(f"  Drawing {len(bboxes)} bounding box(es)...")
                
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    # Draw bounding box
                    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, image.width))
                    y1 = max(0, min(y1, image.height))
                    x2 = max(0, min(x2, image.width))
                    y2 = max(0, min(y2, image.height))
                    
                    # Draw rectangle with red border
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
                    
                    # Draw label background and text
                    try:
                        # Try to get a font
                        try:
                            font = ImageFont.truetype("arial.ttf", 24)
                        except:
                            font = ImageFont.load_default()
                    except:
                        font = None
                    
                    # Draw label text
                    label_text = f"{label}"
                    if font:
                        bbox_text = draw.textbbox((x1, y1 - 30), label_text, font=font)
                    else:
                        bbox_text = draw.textbbox((x1, y1 - 15), label_text)
                    
                    # Draw background for text
                    draw.rectangle(bbox_text, fill='red', outline='red')
                    draw.text((x1, y1 - 30 if font else y1 - 15), label_text, fill='white', font=font)
                    
                    print(f"    Object {i+1}: {label} at [{x1}, {y1}, {x2}, {y2}]")
            
            # Determine output path
            if output_path is None:
                output_path = str(Path(image_path).parent / (
                    Path(image_path).stem + "_florence2_detections.png"
                ))
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save annotated image
            annotated_image.save(output_path)
            print(f"\n[OK] Annotated image saved to: {output_path}")
            print(f"     Full path: {Path(output_path).absolute()}")
            
        except Exception as e:
            print(f"WARNING: Could not create visualization: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nNote: Detection results were still successful - see output above.")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Florence-2 object detection on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on a sample image
  python test_florence2_detection.py --image-path test_image.png
  
  # Use Florence-2-large instead (requires flash_attn - may not work on Windows)
  python test_florence2_detection.py --image-path test_image.png --model-id microsoft/Florence-2-large
  
  # Specify output path
  python test_florence2_detection.py --image-path test_image.png --output-path results.png
        """
    )
    parser.add_argument("--image-path", "-i", type=str, required=True,
                       help="Path to the image file to test")
    parser.add_argument("--model-id", type=str, default="microsoft/Florence-2-base-ft",
                       help="Florence-2 model ID (default: microsoft/Florence-2-base-ft)")
    parser.add_argument("--output-path", "-o", type=str, default=None,
                       help="Output path for annotated image (default: input_name_florence2_detections.png)")
    parser.add_argument("--cache-dir", type=str, 
                       default=r"C:\Users\brian\OneDrive\Desktop\dataprocessing\florence2_models",
                       help="Directory where model is cached (default: florence2_models in project root)")
    
    args = parser.parse_args()
    
    success = test_object_detection(
        args.image_path,
        args.model_id,
        args.output_path,
        args.cache_dir
    )
    
    if success:
        print("\n[OK] Test completed successfully!")
        return 0
    else:
        print("\n[ERROR] Test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

