import wandb
from ultralytics import YOLO
import torch

# Eigene Trainings- oder Evaluierungskonfiguration
toggle = 't'  # 't' für Training, 'e' für Nutzung des trainierten Modells
img_path = 'img'  # Verzeichnis zum Speichern der Ergebnisbilder
txt_path = 'txt'  # Verzeichnis zum Speichern der Ergebnistextdateien

# Liste von Trainingskonfigurationen
sweep_configuration = {
    'name': 'master_sweep',
    'method': 'random',
    "metric": {"goal": "maximize", "name": "val_acc"},
    'parameters': {
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'batch': {'values': [2, 4, 8, 16]},
        'lr0': {'values': [0.001, 0.005, 0.01]},
        'hsv_h': {'values': [0.01, 0.02, 0.03, 0.04, 0.05]},
        'hsv_s': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'hsv_v': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'imgsz': {'values': [416, 512, 608, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280]},
        'lrf': {'values': [0.001, 0.005, 0.01, 0.05, 0.1]},
        'momentum': {'values': [0.9, 0.92, 0.94, 0.96, 0.98]},
        'weight_decay': {'values': [0.0001, 0.0005, 0.001, 0.005]},
        'warmup_epochs': {'values': [1, 2, 3, 4, 5]},
        'warmup_momentum': {'values': [0.6, 0.7, 0.8, 0.9]},
        'warmup_bias_lr': {'values': [0.05, 0.1, 0.2, 0.3]},
        'box': {'values': [5.0, 7.5, 10.0, 12.5, 15.0]},
        'cls': {'values': [0.3, 0.5, 0.7, 0.9]},
        'dfl': {'values': [1.0, 1.5, 2.0, 2.5]},
        'pose': {'values': [8.0, 10.0, 12.0, 14.0]},
        'kobj': {'values': [0.8, 1.0, 1.2, 1.4]},
        'label_smoothing': {'values': [0.0, 0.1, 0.2, 0.3]},
        'nbs': {'values': [16, 32, 64, 128]},
        'degrees': {'values': [0, 15, 30, 45, 60]},
        'translate': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'scale': {'values': [0.5, 0.75, 1.0]},
        'shear': {'values': [0, 10, 20, 30, 40]},
        'perspective': {'values': [0, 0.1, 0.2, 0.3, 0.4]},
        'flipud': {'values': [0, 0.25, 0.5, 0.75, 1.0]},
        'fliplr': {'values': [0, 0.25, 0.5, 0.75, 1.0]},
        'mosaic': {'values': [0, 0.25, 0.5, 0.75, 1.0]},
        'mixup': {'values': [0, 0.25, 0.5, 0.75, 1.0]},
        'copy_paste': {'values': [0, 0.25, 0.5, 0.75, 1.0]},
        'auto_augment': {'values': ['randaugment', 'autoaugment']},
        'erasing': {'values': [0.0, 0.1, 0.2, 0.3, 0.4]},
        'crop_fraction': {'values': [0.5, 0.75, 1.0]}
    }
}

# Basis-Konfiguration für YOLO
base_config = {
    'model': '/workspace/main_folder/models/yolov8x-obb.pt',
    'data': 'Roewaplan.yaml',
    'epochs': 1,
    'save': True,
    'pretrained': True,
    'optimizer': 'auto',
    'project': '/workspace/Yolov8/results/train',
}

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def main():
    wandb.init()
    
    # Erstelle eine kombinierte Konfiguration für das Training
    train_config = {**base_config}
    train_config.update({
        'dropout': wandb.config.dropout,
        'batch': wandb.config.batch,
        'lr0': wandb.config.lr0,
        'hsv_h': wandb.config.hsv_h,
        'hsv_s': wandb.config.hsv_s,
        'hsv_v': wandb.config.hsv_v,
        'imgsz': wandb.config.imgsz,
        'lrf': wandb.config.lrf,
        'momentum': wandb.config.momentum,
        'weight_decay': wandb.config.weight_decay,
        'warmup_epochs': wandb.config.warmup_epochs,
        'warmup_momentum': wandb.config.warmup_momentum,
        'warmup_bias_lr': wandb.config.warmup_bias_lr,
        'box': wandb.config.box,
        'cls': wandb.config.cls,
        'dfl': wandb.config.dfl,
        'pose': wandb.config.pose,
        'kobj': wandb.config.kobj,
        'label_smoothing': wandb.config.label_smoothing,
        'nbs': wandb.config.nbs,
        'degrees': wandb.config.degrees,
        'translate': wandb.config.translate,
        'scale': wandb.config.scale,
        'shear': wandb.config.shear,
        'perspective': wandb.config.perspective,
        'flipud': wandb.config.flipud,
        'fliplr': wandb.config.fliplr,
        'mosaic': wandb.config.mosaic,
        'mixup': wandb.config.mixup,
        'copy_paste': wandb.config.copy_paste,
        'auto_augment': wandb.config.auto_augment,
        'erasing': wandb.config.erasing,
        'crop_fraction': wandb.config.crop_fraction,
        }
    )
    
    # Initialisiere das YOLO-Modell
    model = YOLO(train_config['model']).to(device)
    
    # Trainiere das Modell
    results = model.train(
        **train_config, 
        device=device
    )

    wandb.log()

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep_home_alabama")

# Start sweep job.
wandb.agent(sweep_id, function=main, count=75)
