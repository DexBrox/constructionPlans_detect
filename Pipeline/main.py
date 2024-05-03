import os

from ultralytics import YOLO
from evaluation_functions import *

image_path = '/workspace/datasets/Roewaplan/images/test'
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
image_files = glob.glob(f'{image_path}/*.jpg')
for image in image_files:
    # YOLO Modell auf Bilder anwenden
    results = process_images(image, model)
    data_list = save_results(results, image, img_folder, txt_folder)

    # Textdateien filtern und speichern
    filter_text_files(data_list)






