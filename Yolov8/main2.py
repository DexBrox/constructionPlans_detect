import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch

# Eigene Trainings- oder Evaluierungskonfiguration
toggle = 't'  # 't' für Training, 'e' für Nutzung des trainierten Modells
img_path = 'img'  # Verzeichnis zum Speichern der Ergebnisbilder
txt_path = 'txt'  # Verzeichnis zum Speichern der Ergebnistextdateien

# Liste von Trainingskonfigurationen
configurations = [
    #{'dropout': 0.1, 'batch': 4,  'hsv_h': 0.01, 'hsv_s': 0.6, 'hsv_v': 0.3, 'imgsz': 640, 'lr0': 0.01},
    #{'dropout': 0.2, 'batch': 8,  'hsv_h': 0.02, 'hsv_s': 0.5, 'hsv_v': 0.2, 'imgsz': 640, 'lr0': 0.005},
   #notdone {'dropout': 0.3, 'batch': 16, 'hsv_h': 0.03, 'hsv_s': 0.7, 'hsv_v': 0.4, 'imgsz': 1280, 'lr0': 0.001},
   # {'dropout': 0.4, 'batch': 2,  'hsv_h': 0.04, 'hsv_s': 0.4, 'hsv_v': 0.1, 'imgsz': 1280, 'lr0': 0.01},
   # {'dropout': 0.5, 'batch': 1,  'hsv_h': 0.05, 'hsv_s': 0.3, 'hsv_v': 0.3, 'imgsz': 640, 'lr0': 0.005},
   # {'dropout': 0.6, 'batch': 4,  'hsv_h': 0.06, 'hsv_s': 0.2, 'hsv_v': 0.2, 'imgsz': 1280, 'lr0': 0.001},
   # {'dropout': 0.7, 'batch': 8,  'hsv_h': 0.07, 'hsv_s': 0.6, 'hsv_v': 0.1, 'imgsz': 640, 'lr0': 0.01},
    {'dropout': 0.8, 'batch': 16, 'hsv_h': 0.08, 'hsv_s': 0.7, 'hsv_v': 0.4, 'imgsz': 1280, 'lr0': 0.005},
    {'dropout': 0.1, 'batch': 2,  'hsv_h': 0.09, 'hsv_s': 0.8, 'hsv_v': 0.3, 'imgsz': 640, 'lr0': 0.001}, # übersprungen
    {'dropout': 0.2, 'batch': 1,  'hsv_h': 0.1,  'hsv_s': 0.4, 'hsv_v': 0.2, 'imgsz': 1280, 'lr0': 0.01},
    {'dropout': 0.3, 'batch': 4,  'hsv_h': 0.11, 'hsv_s': 0.3, 'hsv_v': 0.1, 'imgsz': 640, 'lr0': 0.005},
    {'dropout': 0.4, 'batch': 8,  'hsv_h': 0.12, 'hsv_s': 0.2, 'hsv_v': 0.3, 'imgsz': 1280, 'lr0': 0.001},
    {'dropout': 0.5, 'batch': 16, 'hsv_h': 0.13, 'hsv_s': 0.1, 'hsv_v': 0.4, 'imgsz': 640, 'lr0': 0.01},
    {'dropout': 0.6, 'batch': 2,  'hsv_h': 0.14, 'hsv_s': 0.9, 'hsv_v': 0.2, 'imgsz': 1280, 'lr0': 0.005},
    {'dropout': 0.7, 'batch': 1,  'hsv_h': 0.15, 'hsv_s': 0.8, 'hsv_v': 0.1, 'imgsz': 640, 'lr0': 0.001},
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

# Basis-Konfiguration für YOLO
base_config = {
    'model': '/workspace/main_folder/models/yolov8x-obb.pt',
    'data': 'Roewaplan.yaml',
    'epochs': 1000,
    'patience': 1000,
    'imgsz': 1280,
    'save': True,
    'pretrained': True,
    'optimizer': 'auto',
    'project': '/workspace/Yolov8/results/train',
}

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

for config in configurations:
    try:
        # Starte einen neuen W&B Run
        wandb.init(project="Masterarbeit", config=config)
        add_wandb_callback( enable_model_checkpointing=True)
        # start a new wandb run to track this script

        # Erstelle eine kombinierte Konfiguration für das Training
        train_config = {**base_config, **config}
        
        # Initialisiere das YOLO-Modell
        model = YOLO(train_config['model']).to(device)
        
        # Trainiere das Modell
        model.train(
            **train_config,  # Spread operator, um alle Schlüssel-Wert-Paare zu übergeben
            device=device
        )

        model.val()
        model.export(format='onnx')

        # Beende den aktuellen W&B Run
        wandb.finish()

    except Exception as e:
        print(f"Fehler beim Training mit Konfiguration: {config}")
        print(f"Fehlermeldung: {e}")

        # Beende den aktuellen W&B Run
        wandb.finish()
        continue



