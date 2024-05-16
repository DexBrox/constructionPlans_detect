import cv2
import os
import glob
import numpy as np

def draw_bounding_boxes(image, texts, points_list, output_folder, base_name):
    height, width = image.shape[:2]

    for text, points_str in zip(texts, points_list):
        points_float = [float(coord) for coord in points_str.split()]
        
        # Konvertieren normierter Koordinaten in Pixelkoordinaten
        points = np.array([(x * width, y * height) for x, y in zip(points_float[::2], points_float[1::2])], np.int32)
        
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)  # Helles Grün für Linien
        cv2.putText(image, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Helles Rot für Text

    output_path = os.path.join(output_folder, f"{base_name}_obb.jpg")
    cv2.imwrite(output_path, image)

image_folder = '/workspace/datasets/Roewaplan/images/train'
label_folder = '/workspace/datasets/Roewaplan/labels/train'
output_folder = 'output'

# Stelle sicher, dass der Ausgabeordner existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iteriere über alle Bilddateien
for image_path in glob.glob(os.path.join(image_folder, '*.jpg')):
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

        print('one img done')
