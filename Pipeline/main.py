from ultralytics import YOLO
import os

from Pipeline.helper_functions.evaluation_functions import *

image_folder = '/workspace/datasets/Roewaplan/images/test'
model_path = '/workspace/main_folder/models/best.pt'
results_folder = '/workspace/Pipeline/results'

# Model laden
model = YOLO(model_path).to('cuda:1')

# Verzeichnisse für Ergebnisse erstellen
img_folder = os.path.join(results_folder, 'images')
txt_folder = os.path.join(results_folder, 'labels')
target_folder = os.path.join(results_folder, 'labels_beschriftung')
os.makedirs(img_folder, exist_ok=True)
os.makedirs(txt_folder, exist_ok=True)
os.makedirs(target_folder, exist_ok=True)

# YOLO Modell auf Bilder anwenden
process_images(image_folder, model, img_folder, txt_folder)

# Textdateien filtern und speichern
filter_text_files(txt_folder, target_folder)



