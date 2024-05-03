import os
from ultralytics import YOLO
from detect_evaluation_functions import *

import pytesseract
from PIL import Image

dataset_path = '/workspace/datasets/Roewaplan/images/test'
model_path = '/workspace/main_folder/models/best.pt'
results_path = '/workspace/main_folder/results'

# Model laden
model = YOLO(model_path).to('cuda:1')

# Verzeichnisse für Ergebnisse erstellen
img_folder = os.path.join(results_path, 'images')
txt_folder = os.path.join(results_path, 'labels')
target_folder = os.path.join(results_path, 'labels_beschriftung')
os.makedirs(img_folder, exist_ok=True)
os.makedirs(txt_folder, exist_ok=True)
os.makedirs(target_folder, exist_ok=True)

# Durchlaufen aller Bilder
image_files = glob.glob(f'{dataset_path}/*.jpg')
for image_path in image_files:
    # YOLO Modell auf Bilder anwenden
    results = process_images(image_path, model)
    data_list = save_results(results, image_path, img_folder, txt_folder)

    # Textdateien filtern und speichern
    filter_text_files(data_list)

    image = Image.open(image_path)
    for data in data_list:
        
        width, height = image.size
        cls, xy1, xy2, xy3, xy4 = data
        x_min, y_min, x_max, y_max = xy1[0]*width, xy1[1]*height, xy3[0]*width, xy3[1]*height
        print(x_min, y_min, x_max, y_max)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        text = pytesseract.image_to_string(cropped_image, lang='deu')  
        print(text)









