from ultralytics import YOLO
import os
import glob
import torch

# GPU-Check und Geräteauswahl
num_cuda_devices = torch.cuda.device_count()
print("Anzahl der verfügbaren CUDA-GPUs:", num_cuda_devices)
print(torch.cuda.is_available())
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Abrufen der Testbilder
image_files = glob.glob('/workspace/datasets/Roewaplan/images/test/*.jpg')

for i in range(1,10):
    print(f'i-Number: {i}')
    if i == 1:
        # Initialisieren des YOLO-Modells und Verschieben auf GPU
        model = YOLO('/workspace/main_folder/models/yolov8m-obb.pt').to(device)
    else:
        model = YOLO(f'/workspace/Yolov8/results/train/train_{i}/weights/best.pt')
        
    # Trainieren des Modells
    model.train(
        data='Roewaplan_semi.yaml',
        dropout=0.3,
        epochs=15,
        batch=1,
        imgsz=1280,
        patience=150,
        save=True,
        pretrained=True,
        optimizer='auto',
        project='/workspace/Yolov8/results/train',
        device=device,
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
        name=f'train_{i}'
    )

    # Modell validieren und exportieren
    model.val()
    model.export(format='onnx')
    for i in range(1,5):
        print('PRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINT')
    print(f"Modell saved as" + f'train_{i}')
    print(f"Using Modell" + f'/workspace/Yolov8/results/train/train_{i}/weights/best.pt')
    # Verwenden des trainierten Modells
    model = YOLO(f'/workspace/Yolov8/results/train/train_{i}/weights/best.pt')

    # Ergebnisse speichern
    for image in image_files:
        results = model.predict([image], conf=0.5)
        for result in results:
            base_filename = os.path.basename(image)
            name, ext = os.path.splitext(base_filename)
            new_filename = f"{name}_d{ext}"
            new_filename_txt = f"{name}.txt"
            folder_txt_train_semi = os.path.join('/workspace/datasets/Roewaplan_semi/labels/train_semi')
            txt_path_full = os.path.join(folder_txt_train_semi, new_filename_txt)
            result.save_txt(txt_path_full)

