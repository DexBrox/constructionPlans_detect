import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Masterarbeit",
)

from ultralytics import YOLO
import os
import glob
import torch

# Anzahl der verfügbaren CUDA-GPUs abrufen
num_cuda_devices = torch.cuda.device_count()
print("Anzahl der verfügbaren CUDA-GPUs:", num_cuda_devices)
print (torch.cuda.is_available())
device = 'cuda:0' 

# Eigene Trainings- oder Evaluierungskonfiguration
toggle = 't'  # 't' für Training, 'e' für Nutzung des trainierten Modells
img_path = 'img'  # Verzeichnis zum Speichern der Ergebnisbilder
txt_path = 'txt'  # Verzeichnis zum Speichern der Ergebnistextdateien

# YOLOv8-Trainingskonfiguration
config = {
    'model': '/workspace/main_folder/models/yolov8x-obb.pt',
    'data': 'Roewaplan.yaml',
    'dropout': 0.3,
    'epochs': 1000,
    'patience': 1000,
    'batch_size': 1,
    'img_size': 1280,
    'save': True,
    'pretrained': True,
    'optimizer': 'auto',
    'project': '/workspace/Yolov8/results/train',
}

# Konfiguration für Datenanreicherung
conifg_aug = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'auto_augment': 'randaugment',
    'erasing': 0.4,
    'crop_fraction': 1.0
}

# Abrufen der Testbilder
image_files = glob.glob('/workspace/datasets/Roewaplan/images/test/*.jpg')

# Trainings- oder Evaluierungsprozess
if toggle == 't':
    # Initialisieren des YOLO-Modells und Verschieben auf GPU
    model = YOLO(config['model']).to(device)

    # Trainieren des Modells
    model.train(
        model=config['model'],
        data=config['data'],
        dropout=config['dropout'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        patience=config['patience'],
        save=config['save'],
        pretrained=config['pretrained'],
        optimizer=config['optimizer'],
        project=config['project'],

        hsv_h=conifg_aug['hsv_h'],
        hsv_s=conifg_aug['hsv_s'],
        hsv_v=conifg_aug['hsv_v'],
        degrees=conifg_aug['degrees'],
        translate=conifg_aug['translate'],
        scale=conifg_aug['scale'],
        shear=conifg_aug['shear'],
        perspective=conifg_aug['perspective'],
        flipud=conifg_aug['flipud'],
        fliplr=conifg_aug['fliplr'],
        bgr=conifg_aug['bgr'],
        mosaic=conifg_aug['mosaic'],
        mixup=conifg_aug['mixup'],
        copy_paste=conifg_aug['copy_paste'],
        auto_augment=conifg_aug['auto_augment'],
        erasing=conifg_aug['erasing'],
        crop_fraction=conifg_aug['crop_fraction'],
        device=device
    )

    model.export(format='onnx')

elif toggle == 'e':
    # Verwenden des trainierten Modells
    model = YOLO('/workspace/Yolov8/results/train/train2/weights/best.pt')

else:
    print("Ungültiger Wert für 'toggle'")
    exit()

# Iteration über jede Bilddatei
for image in image_files:
    results = model.predict([image], conf=0.15)
    # Speichern der Ergebnisbilder und Textdateien
    for result in results:
        base_filename = os.path.basename(image)
        name, ext = os.path.splitext(base_filename)    
        new_filename = f"{name}_d{ext}"
        new_filename_txt = f"{name}.txt"
        img_folder = os.path.join('/workspace/Yolov8/results', img_path)
        txt_folder = os.path.join('/workspace/Yolov8/results', txt_path)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        if not os.path.exists(txt_folder):
            os.makedirs(txt_folder)
        img_path_full = os.path.join(img_folder, new_filename)
        txt_path_full = os.path.join(txt_folder, new_filename_txt)

        result.save(img_path_full, font_size=25)
        result.save_txt(txt_path_full)





