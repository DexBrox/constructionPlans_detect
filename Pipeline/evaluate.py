import os
import numpy as np
from sklearn.metrics import precision_score

def read_labels(file_path):
    with open(file_path, 'r') as file:
        labels = []
        for line in file:
            parts = line.strip().split()
            if len(parts) > 0:
                label = parts[0]
                bbox = list(map(float, parts[1:9]))
                labels.append((label, bbox))
        return labels

def calculate_iou(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_bboxes(true_labels, pred_labels, iou_threshold=0.95):
    matched_labels = []
    used_preds = set()
    for true_label, true_bbox in true_labels:
        match_found = False
        for i, (pred_label, pred_bbox) in enumerate(pred_labels):
            if i in used_preds:
                continue
            if true_label == pred_label:
                iou = calculate_iou(true_bbox, pred_bbox)
                if iou >= iou_threshold:
                    matched_labels.append(pred_label)
                    used_preds.add(i)
                    match_found = True
                    break
        if not match_found:
            matched_labels.append(None)  # Keine Übereinstimmung gefunden
    return matched_labels

def calculate_precision(true_labels, pred_labels):
    matched_labels = match_bboxes(true_labels, pred_labels)
    y_true = [label for label, _ in true_labels]
    y_pred = [label if label is not None else "None" for label in matched_labels]
    all_classes = set(y_true + y_pred)
    precision_dict = {}
    for cls in all_classes:
        y_true_binary = [1 if label == cls else 0 for label in y_true]
        print(y_true_binary)
        y_pred_binary = [1 if label == cls else 0 for label in y_pred]
        print(y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        precision_dict[cls] = precision
    return precision_dict

def process_files(true_dir, pred_dir):
    all_files = os.listdir(true_dir)
    result = {}
    for file_name in all_files:
        print(f"Processing file {file_name}")
        true_file_path = os.path.join(true_dir, file_name)
        pred_file_path = os.path.join(pred_dir, file_name)
        
        if os.path.exists(pred_file_path):
            true_labels = read_labels(true_file_path)
            pred_labels = read_labels(pred_file_path)
            precision_dict = calculate_precision(true_labels, pred_labels)
            result[file_name] = precision_dict
    return result

# Ordnerpfade anpassen
true_labels_dir = "/workspace/datasets/standard/Roewaplan_v3/labels/test"
predicted_labels_dir = "/workspace/main_folder/PIPELINE_FINAL/results_rp_v3/labels"

precision_results = process_files(true_labels_dir, predicted_labels_dir)

# Ergebnis ausgeben
for file_name, precision_dict in precision_results.items():
    print(f"Datei: {file_name}")
    for cls, precision in precision_dict.items():
        print(f"  Klasse: {cls}, Präzision: {precision:.4f}")
