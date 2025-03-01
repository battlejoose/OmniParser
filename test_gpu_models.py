import torch
import os
from util.utils import get_yolo_model, get_caption_model_processor
from PIL import Image
import numpy as np
import time

# Test GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

# Create necessary directories
if not os.path.exists('temp'):
    os.makedirs('temp', exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), 'temp')
if not os.path.exists('imgs'):
    os.makedirs('imgs', exist_ok=True)

# Test model loading one by one
print("\nTesting YOLO model loading...")
try:
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    print("YOLO model loaded successfully")
    # Force YOLO model to use CUDA
    if torch.cuda.is_available():
        yolo_model.to('cuda')
    print(f"YOLO model device: {yolo_model.device}")
except Exception as e:
    print(f"YOLO model loading failed: {e}")

print("\nTesting caption model loading...")
try:
    caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence/icon_caption")
    print("Caption model loaded successfully")
    print(f"Caption model device: {caption_model_processor['model'].device}")
except Exception as e:
    print(f"Caption model loading failed: {e}")
    import traceback
    traceback.print_exc()

# Create a sample image for testing
test_image = Image.new('RGB', (640, 640), color='white')
test_image_path = os.path.join('imgs', 'test_image.png')
test_image.save(test_image_path)

# Test YOLO model inference
print("\nTesting YOLO model inference...")
try:
    start_time = time.time()
    results = yolo_model.predict(test_image_path, conf=0.05)
    end_time = time.time()
    print(f"YOLO model inference time: {end_time - start_time:.4f} seconds")
    print(f"Number of detections: {len(results[0].boxes)}")
except Exception as e:
    print(f"YOLO model inference failed: {e}")
    import traceback
    traceback.print_exc() 