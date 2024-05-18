import wandb
from ultralytics import YOLO
import os
import glob
import torch
import shutil as sh

# Lösche Train-Ordner
if os.path.exists('/workspace/Yolov8/results_3'):
    sh.rmtree('/workspace/Yolov8/results_3')

# GPU-Check und Geräteauswahl
num_cuda_devices = torch.cuda.device_count()
print("Anzahl der verfügbaren CUDA-GPUs:", num_cuda_devices)
print(torch.cuda.is_available())
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Abrufen der Testbilder
image_files = glob.glob('/workspace/datasets/Roewaplan/images/test/*.jpg')

for i in range(1,15):
    wandb.init(
    project="Masterarbeit_semisupervised_3",
    )
    print(f'i-Number: {i}')
    if i == 1:
        # Initialisieren des YOLO-Modells und Verschieben auf GPU
        model = YOLO('/workspace/main_folder/models/yolov8x-obb.pt').to(device)
    else:
        model = YOLO('/workspace/main_folder/models/yolov8x-obb.pt').to(device)
        #model = YOLO(f'/workspace/Yolov8/results/train_{i-1}/weights/best.pt').to(device)
        
    # Trainieren des Modells
    model.train(
        data='Roewaplan_semi.yaml',
        dropout=0.3,
        epochs=300,
        batch=1,
        imgsz=1280,
        patience=1000,
        save=True,
        pretrained=True,
        optimizer='auto',
        project='results_3',
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
    model.export(format='onnx')
    for k in range(1,5):
        print('PRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINTPRINT')
    print(f"Modell saved as " + f'train_{i}')
    print(f"Using Modell " + f'/workspace/Yolov8/results/train_{i}/weights/best.pt')

    # Verwenden des trainierten Modells
    model = YOLO('/workspace/main_folder/models/yolov8x-obb.pt').to(device)
    #model = YOLO(f'/workspace/Yolov8/results/train_{i}/weights/best.pt')

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
            print(txt_path_full)
            result.save_txt(txt_path_full)

    # Pfade definieren
    label_dir = '/workspace/datasets/Roewaplan_semi/labels/train_semi'
    image_src_dir = '/workspace/datasets/Roewaplan_semi/images/test'
    image_dst_dir = '/workspace/datasets/Roewaplan_semi/images/train_semi'

    # Sicherstellen, dass das Zielverzeichnis existiert
    if not os.path.exists(image_dst_dir):
        os.makedirs(image_dst_dir)

    # Alle Labeldateien im Labelordner finden
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # Namen der Labeldateien ohne Erweiterung extrahieren
    label_names = [os.path.splitext(f)[0] for f in label_files]

    # Über alle Labelnamen iterieren und entsprechende Bilder verschieben
    for label_name in label_names:
        image_name = label_name + '.jpg'
        print(image_name)
        image_src_path = os.path.join(image_src_dir, image_name)
        image_dst_path = os.path.join(image_dst_dir, image_name)

        if os.path.exists(image_src_path):
            sh.copy(image_src_path, image_dst_path)
            print(f"Kopiere {image_src_path} nach {image_dst_path}")
        else:
            print(f"Bild {image_src_path} nicht gefunden")

    wandb.finish()

