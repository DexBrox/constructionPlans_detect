import os
from ultralytics import YOLO
from detect_evaluation_functions import *

import pytesseract
from PIL import Image
from tqdm import tqdm  # Import von tqdm

import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

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
image_files = image_files#[4:10]
for image_path in tqdm(image_files, desc="Processing Images"):

    # YOLO Modell auf Bilder anwenden
    results = process_images(image_path, model)
    data_list = save_results(results, image_path, img_folder, txt_folder)

    # Textdateien filtern und speichern
    data_list_filter = filter_text_files(data_list)

    image = Image.open(image_path)
    for data in data_list_filter:

        width, height = image.size
        cls, xy1, xy2, xy3, xy4 = data
        x1, x2, x3, x4 = xy1[0]*width, xy2[0]*width, xy3[0]*width, xy4[0]*width
        y1, y2, y3, y4 = xy1[1]*height, xy2[1]*height, xy3[1]*height, xy4[1]*height
        buffer = 0.02
        x_min, y_min, x_max, y_max = min(x1, x2, x3, x4)*(1-buffer), min(y1, y2, y3, y4)*(1-buffer), max(x1, x2, x3, x4)*(1+buffer), max(y1, y2, y3, y4)*(1+buffer)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        # OCR with pytesseract
        conf = 0.0
        config = '--oem 3 --psm 6 -l deu'

        text = pytesseract.image_to_string(cropped_image, config=config, output_type=pytesseract.Output.STRING)

        # print(x_min, y_min, x_max, y_max)
        # print(image_path)
        # print(text)

print('Ende')







