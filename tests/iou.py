import os
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
                bounding_boxes.append([xmin, ymin, xmax, ymax])
    return bounding_boxes

def merge_overlapping_boxes(boxes):
    """ Verschmelzt alle sich überlappenden Bounding Boxes zu einer Box. """
    if not boxes:
        return []
    boxes.sort(key=lambda x: x[0])  # Sortieren nach xmin
    merged = [boxes[0]]
    for current in boxes[1:]:
        previous = merged[-1]
        if current[0] <= previous[2] and current[1] <= previous[3] and current[3] >= previous[1]:
            merged[-1] = [
                min(previous[0], current[0]),
                min(previous[1], current[1]),
                max(previous[2], current[2]),
                max(previous[3], current[3])
            ]
        else:
            merged.append(current)
    return merged

def compute_iou(box1, box2):
    """ Berechnet die IoU zwischen zwei Bounding Boxes. """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x1_inter < x2_inter and y1_inter < y2_inter:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection)
    return 0

def compare_folders(ground_truth_dir, predictions_dir):
    results = {}
    gt_files = {f: os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir)}
    pred_files = {f: os.path.join(predictions_dir, f) for f in os.listdir(predictions_dir)}

    for file_name in gt_files.keys():
        if file_name in pred_files and not file_name.startswith('Vertikalschnitt Fußpunkt mit Rinne'):
            gt_boxes = parse_bounding_boxes(gt_files[file_name])
            pred_boxes = merge_overlapping_boxes(parse_bounding_boxes(pred_files[file_name]))
            matched_pred = set()
            ious = []

            for i, gt_box in enumerate(gt_boxes):
                match_found = False
                for j, pred_box in enumerate(pred_boxes):
                    if j not in matched_pred:
                        iou = compute_iou(gt_box, pred_box)
                        if iou > 0:
                            matched_pred.add(j)
                            match_found = True
                            ious.append(iou)
                if not match_found:
                    ious.append(0)  # Keine Übereinstimmung für diese GT-Box

            unmatched_pred = len(pred_boxes) - len(matched_pred)
            unmatched_gt = len(gt_boxes) - sum(1 for iou in ious if iou > 0)

            results[file_name] = {
                'average_iou': np.mean(ious) if ious else 0,
                'total_gt_boxes': len(gt_boxes),
                'total_pred_boxes': len(pred_boxes),
                'matched_boxes': len(matched_pred),
                'unmatched_gt_boxes': unmatched_gt,
                'unmatched_pred_boxes': unmatched_pred
            }

    return results

# Beispiel für den Gebrauch der Funktion:
ground_truth_dir = '/workspace/datasets/standard/Roewaplan_v3/labels/test'
predictions_dir = '/workspace/OCR/results_rpv3/txt/tesseract1'
results = compare_folders(ground_truth_dir, predictions_dir)

for file_name, data in results.items():
    print(f'File: {file_name}')
    print(f'Average IoU: {data["average_iou"]}')
    print(f'Total GT Boxes: {data["total_gt_boxes"]}')
    print(f'Total Pred Boxes: {data["total_pred_boxes"]}')
    print(f'Matched Boxes: {data["matched_boxes"]}')
    print(f'Unmatched GT Boxes: {data["unmatched_gt_boxes"]}')
    print(f'Unmatched Pred Boxes: {data["unmatched_pred_boxes"]}')
    print('-' * 30)
