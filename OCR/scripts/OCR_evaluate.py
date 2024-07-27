import numpy as np
import pandas as pd
import os
import csv

def load_files(gt_file, pred_file):
    """Lädt und verarbeitet GT- und Vorhersagedateien."""
    def process_file(file_path):
        processed_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 8: 
                    text = ' '.join(parts[9:])
                    coords = ' '.join(parts[1:9])  
                    processed_line = f"{coords} {text}"  
                    processed_lines.append(processed_line)
        return processed_lines
    gt_processed = process_file(gt_file)
    pred_processed = process_file(pred_file)
    return gt_processed, pred_processed

def calculate_polygon(gt_processed):
    """Berechnet Polygone aus verarbeiteten Zeilen."""
    polygons = []
    poly_only_text = []
    for line in gt_processed:
        parts = line.split() 
        text = ' '.join(parts[8:])
        coordinates = parts[0:8]
        polygon = [(float(coordinates[i]), float(coordinates[i+1])) for i in range(0, len(coordinates), 2)]
        polygons.append(polygon)
        poly_only_text.append(text)
    return polygons, poly_only_text

def calculate_midpoint(input_lines):
    """Berechnet Mittelpunkte von Polygonen."""
    midpoints = []
    midpoints_w_t = []
    for line in input_lines:
        parts = line.split()
        coordinates = parts[0:8]
        text = ' '.join(parts[8:])
        polygon = [(float(coordinates[i]), float(coordinates[i+1])) for i in range(0, len(coordinates), 2)]
        x_values = [p[0] for p in polygon]
        y_values = [p[1] for p in polygon]
        midpoint = (np.mean(x_values), np.mean(y_values))
        midpoints.append(midpoint)
        midpoints_w_t.append((midpoint, text))
    return midpoints, midpoints_w_t

def point_in_polygon(points, polygon):
    """Bestimmt, ob Punkte innerhalb eines Polygons liegen."""
    all_inside_status = []
    for point in points:
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        all_inside_status.append(inside)
    return all_inside_status

def link_polygons_to_midpoints(polygons, poly_only_text, midpoints, midpoints_w_t):
    """Verknüpft Polygone mit ihren Mittelpunkten."""
    linked_data = []
    df_data = []
    for polygon, poly_text in zip(polygons, poly_only_text):
        for midpoint, mid_text in midpoints_w_t:
            if point_in_polygon([midpoint], polygon)[0]:
                linked_data.append(((polygon, poly_text), midpoint, mid_text))
                df_data.append([str(polygon), poly_text, midpoint[0], midpoint[1], mid_text])
    filtered_linked_data = [data for data in linked_data if data[0][1].strip()]
    return filtered_linked_data

def save_linked_data_to_csv(grouped_data, output_file_path):
    """Speichert verknüpfte Daten in einer CSV-Datei."""
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Polygon', 'Poly Text', 'Midpoint X', 'Midpoint Y', 'Mid Text'])
        for group in grouped_data:
            for entry in group:
                ((polygon, poly_text), (mid_x, mid_y), mid_text) = entry
                polygon_str = ' '.join(f'({x}, {y})' for x, y in polygon)
                writer.writerow([polygon_str, poly_text, mid_x, mid_y, mid_text])

def sort_linked_data_by_polygon_and_midpoint_x(linked_data, y_threshold):
    """Sortiert verknüpfte Daten nach Polygon und Mittelpunkt X-Achse."""
    def primary_sort_key(entry):
        _, midpoint, _ = entry
        return (midpoint[1], midpoint[0])
    sorted_data = sorted(linked_data, key=primary_sort_key)
    def group_data(sorted_data, y_threshold):
        result = []
        current_line = []
        if not sorted_data:
            return result
        current_y = sorted_data[0][1][1]
        for entry in sorted_data:
            _, (midpoint_x, midpoint_y), _ = entry
            if abs(midpoint_y - current_y) > y_threshold:
                if current_line:
                    current_line.sort(key=lambda x: x[1][0])
                    result.append(current_line)
                current_line = [entry]
                current_y = midpoint_y
            else:
                current_line.append(entry)
        if current_line:
            current_line.sort(key=lambda x: x[1][0])
            result.append(current_line)
        return result
    grouped_data = group_data(sorted_data, y_threshold)
    return grouped_data

def aggregate_sum_data(sum_data):
    """Aggregiert Textdaten nach GT-Zeilen."""
    aggregated_dict = {}
    for poly_label, text in sum_data:
        if poly_label in aggregated_dict:
            aggregated_dict[poly_label] += " " + text
        else:
            aggregated_dict[poly_label] = text
    aggregated_data = [(label, text) for label, text in aggregated_dict.items()]
    return aggregated_data

def sum_sentences(sorted_data, i):
    """Summiert Sätze basierend auf sortierten Daten."""
    sum_data = []
    current_first_part_of_polygon = None
    collected_text = ""
    for index, item in enumerate(sorted_data):
        for sub_item in item:
            polygon_with_text = sub_item[0]
            first_part_of_polygon, polygon_text = polygon_with_text[1], sub_item[2]
            if first_part_of_polygon == current_first_part_of_polygon:
                collected_text += " " + polygon_text
            else:
                if current_first_part_of_polygon is not None:
                    sum_data.append((current_first_part_of_polygon, collected_text.strip()))
                current_first_part_of_polygon = first_part_of_polygon
                collected_text = polygon_text if isinstance(polygon_text, str) else str(polygon_text)
    if current_first_part_of_polygon is not None and isinstance(collected_text, str):
        sum_data.append((current_first_part_of_polygon, collected_text.strip()))
    final_sum_data = aggregate_sum_data(sum_data)
    csv_file_path = f'../results/tesseract1/sum_data_{i}.csv'
    df = pd.DataFrame(final_sum_data, columns=['GT Label', 'Predicted Text'])
    directory, filename = os.path.split(csv_file_path)
    os.makedirs(directory, exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    return final_sum_data

def save_sum_data_to_csv(sum_data, output_file_path):
    """Speichert aggregierte Summendaten in einer CSV-Datei."""
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['GT Label', 'Predicted Text'])
        for label, text in sum_data:
            writer.writerow([label, text])
