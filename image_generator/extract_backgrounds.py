import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Ordnerpfade
image_type = '.png'
folder_get_images_train = f'/workspace/datasets/standard/Roewaplan_v3/images/train/*{image_type}'
folder_get_images_val = f'/workspace/datasets/standard/Roewaplan_v3/images/val/*{image_type}'
folder_get_images_test= f'/workspace/datasets/standard/Roewaplan_v3/images/test/*{image_type}'
output_folder = '/workspace/image_generator/background_rp_v3/'

# Sicherstellen, dass der Ausgabeordner existiert
os.makedirs(output_folder, exist_ok=True)

# Funktion, um Bounding Boxes aus einer Label-Datei zu lesen
def read_labels(label_file):
    boxes = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 9:
                # Annahme: Angepasstes Format (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
                class_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts)
                boxes.append((class_id, x1, y1, x2, y2, x3, y3, x4, y4))
    return boxes

# Funktion, um eine Bounding Box in Pixelkoordinaten zu konvertieren
def custom_to_bbox(image_shape, box):
    height, width = image_shape[:2]
    class_id, x1, y1, x2, y2, x3, y3, x4, y4 = box
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    x3 = int(x3 * width)
    y3 = int(y3 * height)
    x4 = int(x4 * width)
    y4 = int(y4 * height)
    # Berechne die Bounding Box, die alle vier Punkte umfasst
    x_min = max(0, min(x1, x2, x3, x4))
    y_min = max(0, min(y1, y2, y3, y4))
    x_max = min(width, max(x1, x2, x3, x4))
    y_max = min(height, max(y1, y2, y3, y4))
    return class_id, x_min, y_min, x_max, y_max

def process_images_for_background(image_folder):
    image_files = glob.glob(image_folder)
    for image_file in tqdm(image_files, desc=f"Processing {os.path.basename(image_folder)}"):
        image = cv2.imread(image_file)
        if image is None:
            print(f"Skipping {image_file} as it could not be read")
            continue
        image_shape = image.shape
        label_file = image_file.replace('/images/', '/labels/').replace(image_type, '.txt')
        if not os.path.exists(label_file):
            print(f"Skipping {image_file} as the corresponding label file does not exist")
            continue
        boxes = read_labels(label_file)
        
        # Weißer Hintergrund über Objekte legen
        for box in boxes:
            class_id, x_min, y_min, x_max, y_max = custom_to_bbox(image_shape, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        
        output_file = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_file, image)

# Verarbeitung von Trainings- und Validierungsordnern
print("Processing training images for background...")
process_images_for_background(folder_get_images_train)
print("Processing validation images for background...")
process_images_for_background(folder_get_images_val)
print("Processing testing images for background...")
process_images_for_background(folder_get_images_test)

print("Hintergründe wurden erfolgreich bereinigt und gespeichert.")
