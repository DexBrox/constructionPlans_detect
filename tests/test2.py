import os
import cv2
import numpy as np

def parse_bounding_boxes(file_path):
    """ Extrahiert Bounding Boxes der Klasse 7 aus einer Datei. """
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == '7':  # Nur Klasse 7 berücksichtigen
                coords = list(map(float, parts[1:9]))
                xmin = min(coords[0], coords[6])
                xmax = max(coords[2], coords[4])
                ymin = min(coords[1], coords[7])
                ymax = max(coords[3], coords[5])
                bounding_boxes.append((xmin, ymin, xmax, ymax))
    return bounding_boxes

def draw_boxes_on_image(image_path, gt_boxes, pred_boxes, output_path):
    """ Zeichnet Ground-Truth und Predicted Bounding Boxes auf das Bild. """
    image = cv2.imread(image_path)
    if image is None:
        return  # Bild konnte nicht geladen werden
    # Zeichne GT Boxes in Grün
    for box in gt_boxes:
        start_point = (int(box[0] * image.shape[1]), int(box[1] * image.shape[0]))
        end_point = (int(box[2] * image.shape[1]), int(box[3] * image.shape[0]))
        image = cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    # Zeichne Pred Boxes in Rot
    for box in pred_boxes:
        start_point = (int(box[0] * image.shape[1]), int(box[1] * image.shape[0]))
        end_point = (int(box[2] * image.shape[1]), int(box[3] * image.shape[0]))
        image = cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)
    # Speichere das Bild
    cv2.imwrite(output_path, image)

def visualize_bboxes(ground_truth_dir, predictions_dir, images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gt_files = {f: os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir)}
    pred_files = {f: os.path.join(predictions_dir, f) for f in os.listdir(predictions_dir)}

    for file_name in gt_files:
        if file_name in pred_files:
            image_path = os.path.join(images_dir, file_name.replace('.txt', '.png'))  # Annahme: Bilder sind .jpg
            if not os.path.exists(image_path):
                continue  # Bild existiert nicht
            gt_boxes = parse_bounding_boxes(gt_files[file_name])
            pred_boxes = parse_bounding_boxes(pred_files[file_name])
            output_path = os.path.join(output_dir, file_name.replace('.txt', '_vis.png'))
            draw_boxes_on_image(image_path, gt_boxes, pred_boxes, output_path)

# Beispiel für den Gebrauch der Funktion:
ground_truth_dir = '/workspace/datasets/standard/Roewaplan_v3/labels/test'
predictions_dir = '/workspace/OCR/results_rpv3/txt/easyocr1'
images_dir = '/workspace/datasets/standard/Roewaplan_v3/images/test'
output_dir = '/workspace/tests/gt_pred_vis/easyocr'
visualize_bboxes(ground_truth_dir, predictions_dir, images_dir, output_dir)
