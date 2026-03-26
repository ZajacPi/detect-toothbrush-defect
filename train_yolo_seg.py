from ultralytics import YOLO

data_yaml = 'dataset.yaml'
pretrained_model = 'yolo26l-seg.pt'

model = YOLO(pretrained_model)

model.train(
    data=data_yaml,
    epochs=300,
    imgsz=320,
    batch=32,
    optimizer='AdamW',
    lr0=0.0001,
    lrf=0.01,
    warmup_epochs=20,

    close_mosaic=0,
    label_smoothing=0.1,
    mask_ratio=1,
    overlap_mask=False,
    val=True,
    box=20.0,
    cls=10.0,
    dfl=2.0,
    copy_paste=0.6,
    mixup=0.1,
    hsv_h=0.015,
    flipud=0.5,
    augment=True
)