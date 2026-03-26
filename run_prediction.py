import os
from pathlib import Path
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

MODEL_PATH = 'runs/segment/train13/weights/best.pt'
INPUT_DIR = 'data/val/images'
OUTPUT_DIR = 'sahi_results_val'
CONF_THRES = 0.25
SLICE_SIZE = 320 
OVERLAP_RATIO = 0.2

detection_model = UltralyticsDetectionModel(
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda:0",
)

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.JPG')
images = [f for f in os.listdir(INPUT_DIR) if f.endswith(image_extensions)]

for img_name in tqdm(images):
    image_path = os.path.join(INPUT_DIR, img_name)
    
    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        verbose=0
    )

    result.export_visuals(
        export_dir=str(output_path),
        file_name=os.path.splitext(img_name)[0] + "_predicted",
        hide_labels=False,
        hide_conf=False
    )

print(f"\nDone! Results saved in: {OUTPUT_DIR}")