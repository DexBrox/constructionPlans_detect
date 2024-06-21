import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Ordnerpfade
image_type = '.png'
folder_get_images_train = f'/workspace/datasets/standard/Roewaplan_v3/images/train/*{image_type}'
folder_get_images_val = f'/workspace/datasets/standard/Roewaplan_v3/images/val/*{image_type}'
folder_get_images_test = f'/workspace/datasets/standard/Roewaplan_v3/images/test/*{image_type}'
output_folder = '/workspace/image_generator/objects_rp_v3/'

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
    return class_id, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

# Dictionary, um die Zähler für jede Klasse zu verfolgen
class_counters = {}

def process_images_and_labels(image_folder):
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
        base_name = os.path.basename(image_file).replace(image_type, '')
        
        for box in boxes:
            class_id, points = custom_to_bbox(image_shape, box)
            points = np.array(points, dtype=np.int32)
            rect = cv2.minAreaRect(points)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            src_pts = box_points.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            # Überprüfen und sicherstellen, dass das Bild 3 Kanäle hat
            if warped.shape[2] == 4:
                warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
            elif warped.shape[2] == 1:
                warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            
            # Initialisieren oder Erhöhen des Zählers für die aktuelle Klasse
            if class_id not in class_counters:
                class_counters[class_id] = 0
            class_counters[class_id] += 1
            # Speichern mit class_id und Index als PNG
            output_file = os.path.join(output_folder, f"{int(class_id)}_{class_counters[class_id]}{image_type}")
            cv2.imwrite(output_file, warped)

# Verarbeitung von Trainings- und Validierungsordnern
print("Processing training images and labels...")
process_images_and_labels(folder_get_images_train)
print("Processing validation images and labels...")
process_images_and_labels(folder_get_images_val)
print("Processing testing images and labels...")
process_images_and_labels(folder_get_images_test)

print("Objekte wurden erfolgreich ausgeschnitten und gespeichert.")
