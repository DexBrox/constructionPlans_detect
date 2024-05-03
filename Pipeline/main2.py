from ultralytics import YOLO
import os
import glob
from PIL import Image
import pytesseract

# Konfiguration der Pfade
image_folder = '/workspace/datasets/Roewaplan/images/test'
model_path = '/workspace/Pipeline/models/best.pt'
results_folder = '/workspace/Pipeline/results'
img_folder = os.path.join(results_folder, 'images')
txt_folder = os.path.join(results_folder, 'labels')
target_folder = os.path.join(results_folder, 'labels_beschriftung')

def process_images(image_folder, model, img_folder, txt_folder):
    image_files = glob.glob(f'{image_folder}/*.jpg')
    for image_path in image_files:
        results = model.predict([image_path], conf=0.15)
        save_results(results, image_path, img_folder, txt_folder)

def save_results(results, image_path, img_folder, txt_folder):
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    new_filename = f"{name}_d{ext}"
    new_filename_txt = f"{name}.txt"
    img_path_full = os.path.join(img_folder, new_filename)
    txt_path_full = os.path.join(txt_folder, new_filename_txt)

    image = Image.open(image_path)
    for result in results:
        result.save(img_path_full, font_size=25)  # Speichern des Ergebnisbildes
        if result.label == 7:  # Klasse 'Beschriftung'
            bbox = result.bbox  # Annahme, dass bbox als Tupel vorliegt
            cropped_image = image.crop(bbox)
            ocr_text = pytesseract.image_to_string(cropped_image)
            with open(txt_path_full, 'a') as f:
                f.write(ocr_text + '\n')

def filter_text_files(txt_folder, target_folder):
    txt_files = glob.glob(f'{txt_folder}/*.txt')
    for file_path in txt_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if 'Beschriftung' in line]
        base_name = os.path.basename(file_path)
        new_file_path = os.path.join(target_folder, base_name)

        with open(new_file_path, 'w') as new_file:
            new_file.writelines(filtered_lines)


# Verzeichnisse erstellen
os.makedirs(img_folder, exist_ok=True)
os.makedirs(txt_folder, exist_ok=True)
os.makedirs(target_folder, exist_ok=True)

# Modell laden und Bilder verarbeiten
model = YOLO(model_path).to('cuda:1')
process_images(image_folder, model, img_folder, txt_folder)

# Textdateien filtern und OCR-Text hinzufügen
filter_text_files(txt_folder, target_folder)

