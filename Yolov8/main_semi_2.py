import wandb
from ultralytics import YOLO
import os
import glob
import torch
import shutil as sh

# Paths
# Folder to save training data
folder_save_train = '/workspace/Yolov8/results_3'
# Folder to load test images
folder_get_images_test = '/workspace/datasets/Roewaplan_semi/images/test/*.jpg'
# Path to load the YOLO model
path_get_model_yolo = '/workspace/main_folder/models/yolov8x-obb.pt'
# Project name for wandb
name_project_wandb = 'Masterarbeit_semisupervised_3_model_x_train_noniterativ'

# Paths for labels and images
label_dir = '/workspace/datasets/Roewaplan_semi/labels/train_semi'
image_src_dir = '/workspace/datasets/Roewaplan_semi/images/test'
image_dst_dir = '/workspace/datasets/Roewaplan_semi/images/train_semi'

# Delete training folder if it exists
if os.path.exists(folder_save_train):
    sh.rmtree(folder_save_train)

# GPU check and device selection
num_cuda_devices = torch.cuda.device_count()
print("Number of available CUDA GPUs:", num_cuda_devices)
print(torch.cuda.is_available())
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def train_model(i, device, path_get_model_yolo):
    # Initialize a wandb run
    with wandb.init(project=name_project_wandb):
        print(f'i-Number: {i}')
        # Initialize the YOLO model and move it to the selected device
        model = YOLO(path_get_model_yolo).to(device)
        
        # Train the model
        model.train(
            data='Roewaplan_semi.yaml',
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
            name=f'train_{i}',
            cache=False,
            workers=8,
        )
        
        # Export the trained model to ONNX format
        model.export(format='onnx')
        print(f"Model saved as train_{i}")
        print(f"Using model /workspace/Yolov8/results/train_{i}/weights/best.pt")
        return model

def predict_and_save_results(model, folder_get_images_test, label_dir):
    # Retrieve test images
    image_files = glob.glob(folder_get_images_test)
    print(image_files)
    
    # Predict and save results for each test image
    for image in image_files:
        results = model.predict([image], conf=0.5)
        for result in results:
            base_filename = os.path.basename(image)
            name, ext = os.path.splitext(base_filename)
            new_filename_txt = f"{name}.txt"
            txt_path_full = os.path.join(label_dir, new_filename_txt)
            print(txt_path_full)
            result.save_txt(txt_path_full)

def move_images(label_dir, image_src_dir, image_dst_dir):
    # Ensure the destination directory exists
    if not os.path.exists(image_dst_dir):
        os.makedirs(image_dst_dir)

    # Find all label files in the label directory
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    # Extract the base names of the label files (without extension)
    label_names = [os.path.splitext(f)[0] for f in label_files]
    
    # Move corresponding images based on label names
    for label_name in label_names:
        image_name = label_name + '.jpg'
        print(image_name)
        image_src_path = os.path.join(image_src_dir, image_name)
        image_dst_path = os.path.join(image_dst_dir, image_name)
        
        if os.path.exists(image_src_path):
            sh.move(image_src_path, image_dst_path)
            print(f"Moving {image_src_path} to {image_dst_path}")
        else:
            print(f"Image {image_src_path} not found")

# Main loop to train the model iteratively
for i in range(1, 15):
    # Train the model and get the trained model instance
    model = train_model(i, device, path_get_model_yolo)
    
    # Predict and save results using the trained model
    predict_and_save_results(model, folder_get_images_test, label_dir)
    
    # Move images based on the generated label files
    move_images(label_dir, image_src_dir, image_dst_dir)
