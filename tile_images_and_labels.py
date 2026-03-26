import cv2
import numpy as np
from pathlib import Path

def tile_images_and_labels(img_dir, label_dir, output_base, tile_size=320, overlap=0.2):
    img_out = Path(output_base) / 'images'
    lbl_out = Path(output_base) / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    stride = int(tile_size * (1 - overlap))
    
    for img_path in Path(img_dir).glob('*.png'):
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        base_name = img_path.stem
        label_path = Path(label_dir) / f"{base_name}.txt"
        
        if not label_path.exists(): continue
        
        with open(label_path, 'r') as f:
            lines = [l.strip().split() for l in f.readlines()]

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                new_labels = []
                for line in lines:
                    cls = line[0]
                    coords = np.array(line[1:], dtype=float).reshape(-1, 2)
                    px_coords = coords * [w, h]
                    mask_in_tile = (px_coords[:, 0] >= x) & (px_coords[:, 0] <= x + tile_size) & \
                                   (px_coords[:, 1] >= y) & (px_coords[:, 1] <= y + tile_size)
                    
                    if np.any(mask_in_tile):
                        rel_coords = (px_coords - [x, y]) / tile_size
                        rel_coords = np.clip(rel_coords, 0, 1)
                        label_str = f"{cls} " + " ".join([f"{c:.6f}" for pair in rel_coords for c in pair])
                        new_labels.append(label_str)

                if new_labels: # Save only tiles with defects
                    tile_name = f"{base_name}_y{y}_x{x}"
                    cv2.imwrite(str(img_out / f"{tile_name}.png"), img[y:y+tile_size, x:x+tile_size])
                    with open(lbl_out / f"{tile_name}.txt", 'w') as f:
                        f.write("\n".join(new_labels))

tile_images_and_labels('data/train/images', 'data/train/labels', 'tiled_data/train')
tile_images_and_labels('data/val/images', 'data/val/labels', 'tiled_data/val')