import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# Funktion zum Zeichnen der Bounding Boxes
def draw_bounding_boxes(image, texts, points_list):
    height, width = image.shape[:2]
    for text, points_str in zip(texts, points_list):
        points_float = [float(coord) for coord in points_str.split()]
        points = np.array([(x * width, y * height) for x, y in zip(points_float[::2], points_float[1::2])], np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Funktion zum Abspeichern der Bounding Boxes in einer Datei
def save_bounding_boxes(label_path, texts, points_list):
    with open(label_path, 'w') as f:
        for text, points_str in zip(texts, points_list):
            f.write(f"{text} {points_str}\n")

# Funktion zur Umwandlung von Pixelkoordinaten in normierte Koordinaten
def normalize_points(points, width, height):
    return [(x / width, y / height) for x, y in points]

# Funktion zur Umwandlung von normierten Koordinaten in Pixelkoordinaten
def denormalize_points(points, width, height):
    return [(x * width, y * height) for x, y in points]

# Globale Variablen zur Speicherung der Klicks und Bounding Boxes
clicks = []
new_texts = []
new_points_list = []

def click_event(event, x, y, flags, param):
    global clicks, new_texts, new_points_list
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        if len(clicks) == 2:
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            new_texts.append('new_label')
            new_points_list.append(f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}")
            clicks = []

def main(image_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in tqdm(glob.glob(os.path.join(image_folder, '*.png'))):
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        label_path = os.path.join(label_folder, f'{name}.txt')

        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            with open(label_path, 'r') as f:
                lines = f.readlines()

            texts = [line.split()[0] for line in lines]
            points_list = [' '.join(line.split()[1:]) for line in lines]

            draw_bounding_boxes(image, texts, points_list)
            cv2.imshow("Image", image)
            cv2.setMouseCallback("Image", click_event)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            cv2.destroyAllWindows()

            # Normiere die neuen Punkte und füge sie den Listen hinzu
            for points_str in new_points_list:
                points_float = [float(coord) for coord in points_str.split()]
                points_norm = normalize_points(points_float, width, height)
                points_list.append(' '.join(map(str, [coord for point in points_norm for coord in point])))
                texts.append('new_label')

            save_bounding_boxes(label_path, texts, points_list)
            output_path = os.path.join(output_folder, f"{name}_obb.jpg")
            cv2.imwrite(output_path, image)

if __name__ == "__main__":
    image_folder = 'rp_v2'
    label_folder = 'label'
    output_folder = 'output'
    main(image_folder, label_folder, output_folder)
