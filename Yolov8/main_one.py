import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os

# Liste von Trainingskonfigurationen
configuration = [
    {
        'dropout': 0.3,
        'batch': 1,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'imgsz': 1280,
        'lr0': 0.01,
        'patience': 1000,
        'cache': False,
        'workers': 8,
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': None,
        'workspace': 4,
        'nms': False,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64
    } 
]


models = ['/workspace/main_folder/models/yolov8n-obb.pt', 
          '/workspace/main_folder/models/yolov8s-obb.pt',
          '/workspace/main_folder/models/yolov8m-obb.pt',
          '/workspace/main_folder/models/yolov8l-obb.pt',
          '/workspace/main_folder/models/yolov8x-obb.pt']

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

for model in models:
    try:
        # Basis-Konfiguration für YOLO
        base_config = {
            'model': '/workspace/main_folder/models/yolov8x-obb.pt',
            'data': 'YAML/Dataset_Theo.yaml',
            'epochs': 1000,
            'patience': 1000,
            'imgsz': 1280,
            'save': True,
            'pretrained': True,
            'optimizer': 'auto',
            'project': '/workspace/Yolov8/results/train',
        }

        model_head, model_tail = os.path.split(model)

        wandb.init(project="Masterarbeit_dataset-theo_model-various", config=configuration, run=model_tail)
        add_wandb_callback(enable_model_checkpointing=True)

        train_config = {**base_config, **configuration}
        model = YOLO(train_config['model']).to(device)
        
        model.train(**train_config, device=device)

        model.val()
        model.export(format='onnx')

        wandb.finish()

    except Exception as e:
        print(f"Fehler beim Training mit Konfiguration: {model}")
        print(f"Fehlermeldung: {e}")

        wandb.finish()
        continue
