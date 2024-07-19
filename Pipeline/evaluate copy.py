import numpy as np
import os

def read_bboxes_from_file(file_path):
    bboxes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            bbox = list(map(float, parts[1:9]))
            bboxes.append(bbox)
    return bboxes

def read_bboxes_from_dir(directory_path):
    bboxes_dict = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        bboxes = read_bboxes_from_file(file_path)
        bboxes_dict[filename] = bboxes
    return bboxes_dict

def bbox_iou(box1, box2):
    def get_area(box):
        return (box[2] - box[0]) * (box[5] - box[1])
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[5], box2[5])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = get_area(box1)
    box2_area = get_area(box2)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def precision_recall(true_bboxes, pred_bboxes, iou_threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched = []
    pairings = []  # Zum Speichern der Paarungen und IoU-Werte
    for pred_box in pred_bboxes:
        best_iou = 0
        best_match = None
        for i, true_box in enumerate(true_bboxes):
            if i in matched:
                continue
            iou = bbox_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_match = i
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched.append(best_match)
            pairings.append((pred_box, true_bboxes[best_match], best_iou))  # Paarung hinzufügen
        else:
            false_positives += 1
        
        # Debugging: IoU-Werte ausgeben
        print(f'Pred Box: {pred_box}, Best IoU: {best_iou}, Threshold: {iou_threshold}')

    false_negatives = len(true_bboxes) - len(matched)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall, pairings

def average_precision(precisions, recalls):
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def mean_average_precision(true_bboxes, pred_bboxes, iou_thresholds):
    ap_values = []
    for iou_threshold in iou_thresholds:
        precisions = []
        recalls = []
        for threshold in np.arange(0.5, 1.0, 0.05):
            precision, recall, _ = precision_recall(true_bboxes, pred_bboxes, threshold)
            precisions.append(precision)
            recalls.append(recall)
        ap = average_precision(precisions, recalls)
        ap_values.append(ap)
    return np.mean(ap_values)

# Dateipfade
true_bboxes_path = '/workspace/datasets/standard/Roewaplan_v3/labels/test'
pred_bboxes_path = '/workspace/main_folder/PIPELINE_FINAL/results_rp_v3/labels'

# BBs laden
true_bboxes_dict = read_bboxes_from_dir(true_bboxes_path)
pred_bboxes_dict = read_bboxes_from_dir(pred_bboxes_path)

# IoU Schwellen
iou_thresholds = np.arange(0.5, 1.0, 0.05)

# mAP und Precision pro Bild berechnen und ausgeben
for filename in true_bboxes_dict:
    true_bboxes = true_bboxes_dict.get(filename, [])
    pred_bboxes = pred_bboxes_dict.get(filename, [])
    
    # Debugging: Anzahl der Bounding Boxen ausgeben
    print(f'{filename}: {len(true_bboxes)} Ground Truth BBs, {len(pred_bboxes)} Predicted BBs')
    
    if len(true_bboxes) == 0 or len(pred_bboxes) == 0:
        print(f'{filename}: mAP@0.50:0.95: 0.0 (Keine BBs)')
        continue
    
    mAP = mean_average_precision(true_bboxes, pred_bboxes, iou_thresholds)
    precision, _, pairings = precision_recall(true_bboxes, pred_bboxes, 0.5)  # Precision bei IoU Schwelle 0.5
    print(f'{filename}: mAP@0.50:0.95: {mAP}, Precision@0.5: {precision}')
    
    # Paarungen und IoU-Werte ausgeben
    for pred_box, true_box, iou in pairings:
        print(f'Pred Box: {pred_box} -> True Box: {true_box}, IoU: {iou}')
        print(f'Zusammengehörige BBs: Pred Box {pred_box}, True Box {true_box}')
