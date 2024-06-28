import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# Definiere Farben für die verschiedenen Klassen (0 bis 15) in BGR
class_colors = {
    '0': (56, 56, 255),    # Blau
    '1': (158, 163, 255),  # Hellblau
    '2': (52, 126, 255),   # Orange
    '3': (42, 182, 255),   
    '4': (67, 214, 211),   # Gelb
    '5': (255, 0, 255),    # Pink
    '6': (255, 0, 255),    # Pink
    '7': (146, 222, 81),   # Hellgrün
    '8': (22, 255, 255),   # Pink
    '9': (255, 115, 100),    
    '10': (49, 155, 170),  # Türkis
    '11': (255, 0, 255),   # Pink
    '12': (147, 69, 52),   
    '13': (255, 115, 100),   
    '14': (255, 0, 255),   # Pink
    '15': (255, 56, 132)    
}

def draw_bounding_boxes(image, texts, points_list, output_folder, base_name):
    height, width = image.shape[:2]

    for text, points_str in zip(texts, points_list):
        points_float = [float(coord) for coord in points_str.split()]
        
        # Konvertieren normierter Koordinaten in Pixelkoordinaten
        points = np.array([(x * width, y * height) for x, y in zip(points_float[::2], points_float[1::2])], np.int32)
        
        # Bestimme die Farbe basierend auf der Klasse
        color = class_colors.get(text, (255, 255, 255))  # Standardfarbe ist Weiß, falls Klasse nicht definiert

        cv2.polylines(image, [points], isClosed=True, color=color, thickness=5)  # Verwende die Farbe für die Linien
        cv2.putText(image, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)  # Verwende die Farbe für den Text

    output_path = os.path.join(output_folder, f"{base_name}_obb.jpg")
    cv2.imwrite(output_path, image)

image_folder = '/workspace/datasets/Theo/images/train'
#image_folder = 'new_labels_13_06/images'
label_folder = image_folder.replace('images', 'labels')
output_folder = 'standard/theo_vis'

# Stelle sicher, dass der Ausgabeordner existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iteriere über alle Bilddateien
for image_path in tqdm(glob.glob(os.path.join(image_folder, '*.jpg'))):
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)
    label_path = os.path.join(label_folder, f'{name}.txt')

    if os.path.exists(label_path):
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        texts = [line.split()[0] for line in lines]
        points_list = [' '.join(line.split()[1:]) for line in lines]

        draw_bounding_boxes(image, texts, points_list, output_folder, name)
