import numpy as np
import pandas as pd
import os
import csv

def load_files(gt_file, pred_file):
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

    # Verarbeite jede Datei und lade die Zeilen
    gt_processed = process_file(gt_file)
    pred_processed = process_file(pred_file)

    return gt_processed, pred_processed

def calculate_polygon(gt_processed):
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
    #print(input_lines)
    midpoints = []
    midpoints_w_t = []

    for line in input_lines:
        parts = line.split()

        # Nehme die letzten 8 Elemente als Koordinaten
        coordinates = parts[0:8]
        # Der Rest ist der Text
        text = ' '.join(parts[8:])

        polygon = [(float(coordinates[i]), float(coordinates[i+1])) for i in range(0, len(coordinates), 2)]

        x_values = [p[0] for p in polygon]
        y_values = [p[1] for p in polygon]
        midpoint = (np.mean(x_values), np.mean(y_values))

        midpoints.append(midpoint)
        midpoints_w_t.append((midpoint, text))

    return midpoints, midpoints_w_t


def point_in_polygon(points, polygon):
    all_inside_status = []
    for point in points:
        x, y = point
        inside = False
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            try:
                xi, yi = polygon[i]
                xj, yj = polygon[j]
            except TypeError:
                print(f"Fehler beim Entpacken der Koordinaten in Polygon: {polygon[i]} oder {polygon[j]}")
                return []

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

        all_inside_status.append(inside)

    return all_inside_status


def link_polygons_to_midpoints(polygons, poly_only_text, midpoints, midpoints_w_t):
    linked_data = []
    df_data = []

    for polygon, poly_text in zip(polygons, poly_only_text):
        for midpoint, mid_text in midpoints_w_t:
            if point_in_polygon([midpoint], polygon)[0]:
                linked_data.append(((polygon, poly_text), midpoint, mid_text))
                df_data.append([str(polygon), poly_text, midpoint[0], midpoint[1], mid_text])

    # Filtere alle Zeilen, die keinen Groundtruth-Text haben
    filtered_linked_data = [data for data in linked_data if data[0][1].strip()]

    save_linked_data_to_csv(filtered_linked_data, 'data_temp.csv')

    return filtered_linked_data

def save_linked_data_to_csv(linked_data, output_file_path):
    """
    Speichert die verknüpften Daten in einer CSV-Datei.

    :param linked_data: Die Liste der verknüpften Daten.
    :param output_file_path: Der Pfad, unter dem die Datei gespeichert werden soll.
    """
    # Öffne die Datei im Schreibmodus
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Schreibe die Kopfzeile
        writer.writerow(['Polygon', 'Poly Text', 'Midpoint X', 'Midpoint Y', 'Mid Text'])
        
        # Schreibe die Datenzeilen
        for ((polygon, poly_text), (mid_x, mid_y), mid_text) in linked_data:
            # Konvertiere Polygon-Punkte in einen String, der in CSV geschrieben werden kann
            polygon_str = ' '.join(f'({x}, {y})' for x, y in polygon)
            writer.writerow([polygon_str, poly_text, mid_x, mid_y, mid_text])

# Beispiel für den Aufruf der Funktion
# save_linked_data_to_csv(linked_data, 'pfad/zur/datei.csv')



#def sort_linked_data_by_polygon_and_midpoint_x(linked_data): #falsch
    """
    Sortiert die verknüpften Daten zunächst basierend auf den vertikalen Werten der Mittelpunkte und innerhalb jeder vertikalen Gruppe nach den horizontalen Werten.
    """
    # Sortieren der Daten basierend auf der vertikalen Mittelpunktkoordinate
    linked_data.sort(key=lambda x: x[1][1])
    
    # Gruppieren nach vertikalem Abstand, hier beispielhaft in 100-Pixel-Schritten
    from itertools import groupby
    grouped_data = groupby(linked_data, key=lambda x: x[1][1] // 10)  # Gruppierung der Daten in vertikalen Schritten
    
    sorted_linked_data = []
    for _, group in grouped_data:
        sorted_group = sorted(group, key=lambda x: x[1][0])  # Sortieren jeder Gruppe nach der horizontalen Mittelpunktkoordinate
        sorted_linked_data.extend(sorted_group)
    
    return sorted_linked_data


def sort_linked_data_by_polygon_and_midpoint_x(linked_data): #old
    def sort_key(entry):
        polygon, (midpoint_x, midpoint_y), _ = entry[0], entry[1], entry[2]
        return (polygon, midpoint_x, -midpoint_y)  

    sorted_data = sorted(linked_data, key=sort_key)

    def custom_sort(sorted_data):
        result = []
        i = 0
        while i < len(sorted_data):
            group = [sorted_data[i]]
            while i + 1 < len(sorted_data) and sorted_data[i][0] == sorted_data[i + 1][0] and abs(sorted_data[i][1][0] - sorted_data[i + 1][1][0]) / sorted_data[i][1][0] < 0.01:
                group.append(sorted_data[i + 1])
                i += 1
            if len(group) > 1:
                group.sort(key=lambda x: x[1][1], reverse=True)
            result.extend(group)
            i += 1
        return result

    sorted_data = custom_sort(sorted_data)
    save_linked_data_to_csv(sorted_data, 'data_temp2.csv')
    return sorted_data


def sum_sentences(sorted_data, i):
    sum_data = []
    
    current_first_part_of_polygon = None
    collected_text = ""

    for item in sorted_data:
        first_part_of_polygon, text = item[0][1], item[2] 

        if first_part_of_polygon == current_first_part_of_polygon:
            collected_text += " " + text
        else:
            if current_first_part_of_polygon is not None:
                sum_data.append((current_first_part_of_polygon, collected_text.strip()))

            current_first_part_of_polygon = first_part_of_polygon
            collected_text = text

    if current_first_part_of_polygon is not None:
        sum_data.append((current_first_part_of_polygon, collected_text.strip()))

    csv_file_path = f'../results/sum_data_{i}.csv'
    df = pd.DataFrame(sum_data, columns=['label', 'predict'])
    directory, filename = os.path.split(csv_file_path)
    os.makedirs(directory, exist_ok=True)
    print (directory)
    df.to_csv(csv_file_path, index=False)

    return sum_data