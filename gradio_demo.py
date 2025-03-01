from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import base64, os
import tempfile
import shutil
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image

# Create a custom temp directory
if not os.path.exists('temp'):
    os.makedirs('temp', exist_ok=True)
# Set custom temp directory
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), 'temp')

# Ensure imgs directory exists
if not os.path.exists('imgs'):
    os.makedirs('imgs', exist_ok=True)

# Print GPU availability for debugging
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Initialize models
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
# Ensure YOLO model uses GPU if available
if torch.cuda.is_available():
    yolo_model.to('cuda')
print(f"YOLO model device: {yolo_model.device}")

caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence/icon_caption")
print(f"Caption model device: {caption_model_processor['model'].device}")

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:

    # Create a safer way to save and access the image
    image_save_path = os.path.join('imgs', 'saved_image_demo.png')
    try:
        # Save with PIL directly instead of relying on image_input.save
        img_pil = Image.fromarray(np.array(image_input))
        img_pil.save(image_save_path)
        image = img_pil
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, f"Error processing image: {e}"
        
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    
    try:
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
        text, ocr_bbox = ocr_bbox_rslt
        
        # Ensure text and ocr_bbox are never None
        if text is None:
            text = []
        if ocr_bbox is None:
            ocr_bbox = []
        
        print(f"Starting image processing with YOLO on {yolo_model.device}")    
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text, iou_threshold=iou_threshold, imgsz=imgsz)  
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        print('finish processing')
        parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
        return image, str(parsed_content_list)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error during processing: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=False)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# Launch the demo with detailed output
print("Starting Gradio server...")
print("Once the server starts, you can access the demo at: http://127.0.0.1:7863")
demo.launch(share=False, server_port=7863, server_name='127.0.0.1')
