import wandb
from ultralytics import YOLO
import torch

for i in range(4,5):

    # Pfade
    # Pfad zum laden des Modells
    path_get_model_yolo = '/workspace/main_folder/models/yolov8x-obb.pt'
    # Projektname für wandb
    name_project_wandb = 'Masterarbeit_datasets_model_x_train_noniterativ'
    run_name = f'Dataset_{i}'

    # GPU-Check und Geräteauswahl
    num_cuda_devices = torch.cuda.device_count()
    print("Anzahl der verfügbaren CUDA-GPUs:", num_cuda_devices)
    print(torch.cuda.is_available())
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    wandb.init(project=name_project_wandb, name=run_name)

    # Initialisieren des YOLO-Modells und Verschieben auf GPU
    model = YOLO(path_get_model_yolo).to(device)

    print(f'YAML/Roewaplan_datasets{i}.yaml')

    # Trainieren des Modells
    model.train(
        data=f'YAML/Roewaplan_data{i}.yaml',
        dropout=0.3,
        batch=4,
        epochs=500,
        imgsz=640,
        patience=1000,
        save=True,
        pretrained=True,
        optimizer='auto',
        project='results_3',
        device=device,

        lr0=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,

        cache = False,
        workers = 8,
    )

    # Modell validieren und exportieren
    model.export(format='onnx')

    wandb.finish()

