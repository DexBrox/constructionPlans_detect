import os
import glob
import cv2
import numpy as np

# Ordnerpfade
image_type = '.png'
folder_get_images_train = f'/workspace/datasets/standard/Roewaplan_org/images/train/*{image_type}'
folder_get_images_val = f'/workspace/datasets/standard/Roewaplan_org/images/val/*{image_type}'
output_folder = '/workspace/image_generator/objekte_white/'

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

# Funktion, um weißen Hintergrund zu entfernen und durch Transparenz zu ersetzen
def remove_white_background(image):
    # Umwandeln in ein 4-Kanal-Bild (RGBA)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    # Alle Pixel im Bereich von 250 bis 255 als transparent (0) setzen
    white = np.all(image[:, :, :3] >= 256, axis=2)
    image_rgba[white, 3] = 0
    return image_rgba

# Dictionary, um die Zähler für jede Klasse zu verfolgen
class_counters = {}

def process_images_and_labels(image_folder):
    for image_file in glob.glob(image_folder):
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
            class_id, x_min, y_min, x_max, y_max = custom_to_bbox(image_shape, box)
            cropped_object = image[y_min:y_max, x_min:x_max]
            # Überprüfen, ob das ausgeschnittene Bild nicht leer ist und nicht das gesamte Bild erfasst
            if cropped_object.size == 0 or cropped_object.shape[0] == image.shape[0] or cropped_object.shape[1] == image.shape[1]:
                print(f"Skipping object in {base_name} due to size issues")
                continue
            # Entfernen des weißen Hintergrunds
            cropped_object = remove_white_background(cropped_object)
            # Initialisieren oder Erhöhen des Zählers für die aktuelle Klasse
            if class_id not in class_counters:
                class_counters[class_id] = 0
            class_counters[class_id] += 1
            # Speichern mit class_id und Index als PNG
            output_file = os.path.join(output_folder, f"{int(class_id)}_{class_counters[class_id]}{image_type}")
            cv2.imwrite(output_file, cropped_object)

# Verarbeitung von Trainings- und Validierungsordnern
print("Processing training images and labels...")
process_images_and_labels(folder_get_images_train)
print("Processing validation images and labels...")
process_images_and_labels(folder_get_images_val)

print("Objekte wurden erfolgreich ausgeschnitten und gespeichert.")
