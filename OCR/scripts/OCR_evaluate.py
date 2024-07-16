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

    #save_linked_data_to_csv(filtered_linked_data, 'data_temp.csv')

    return filtered_linked_data

def save_linked_data_to_csv(grouped_data, output_file_path):
    """
    Speichert die verknüpften Daten in einer CSV-Datei. Geht davon aus, dass `grouped_data`
    eine Liste von Listen ist, wobei jede innere Liste Tupel von Daten enthält.

    :param grouped_data: Die Liste der verknüpften Daten.
    :param output_file_path: Der Pfad, unter dem die Datei gespeichert werden soll.
    """
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Schreibe die Kopfzeile
        writer.writerow(['Polygon', 'Poly Text', 'Midpoint X', 'Midpoint Y', 'Mid Text'])

        for group in grouped_data:  # Gehe durch jede Gruppe in den verknüpften Daten
            for entry in group:
                if len(entry) != 3:
                    print(f"Unerwartete Eintragslänge: {len(entry)} für Eintrag: {entry}")
                    continue  # Überspringe diesen Eintrag

                try:
                    # Versucht, jedes Tupel (Polygon, Text), Mittelpunkt, Text zu entpacken
                    ((polygon, poly_text), (mid_x, mid_y), mid_text) = entry
                    # Konvertiere Polygon-Punkte in einen String, der in CSV geschrieben werden kann
                    polygon_str = ' '.join(f'({x}, {y})' for x, y in polygon)
                    writer.writerow([polygon_str, poly_text, mid_x, mid_y, mid_text])
                except Exception as e:
                    print(f"Fehler beim Verarbeiten eines Eintrags: {e}")


def sort_linked_data_by_polygon_and_midpoint_x(linked_data, y_threshold=0.01):
    def primary_sort_key(entry):
        _, midpoint, _ = entry
        return (midpoint[1], midpoint[0])

    sorted_data = sorted(linked_data, key=primary_sort_key)

    def group_data(sorted_data, y_threshold):
        result = []
        current_line = []

        if not sorted_data:
            return result

        # Startbedingung für die erste Zeile
        current_y = sorted_data[0][1][1]

        for entry in sorted_data:
            _, (midpoint_x, midpoint_y), _ = entry
            # Prüfe, ob ein neuer Zeilenwechsel vorliegt
            if abs(midpoint_y - current_y) > y_threshold:
                if current_line:
                    # Sortiere die aktuelle Zeile nach X-Koordinate vor dem Hinzufügen zum Ergebnis
                    current_line.sort(key=lambda x: x[1][0])
                    result.append(current_line)
                current_line = [entry]
                current_y = midpoint_y
            else:
                current_line.append(entry)

        # Füge die letzte Zeile hinzu und sortiere sie, falls vorhanden
        if current_line:
            current_line.sort(key=lambda x: x[1][0])
            result.append(current_line)
        return result

    grouped_data = group_data(sorted_data, y_threshold)
    #save_linked_data_to_csv(grouped_data, 'temp_data_hallo.csv')
    return grouped_data

#def sort_linked_data_by_polygon_and_midpoint_x(linked_data): #old
    def sort_key(entry):
        polygon, (midpoint_x, midpoint_y), _ = entry[0], entry[1], entry[2]
        return (polygon, midpoint_x, -midpoint_y)  
    
    #print(linked_data)

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
    print(sorted_data)
    return sorted_data

def aggregate_sum_data(sum_data):
    # Verwende ein Dictionary, um die Texte basierend auf den GT-Zeilen zu konsolidieren
    aggregated_dict = {}
    for poly_label, text in sum_data:
        if poly_label in aggregated_dict:
            # Füge den aktuellen Text zum vorhandenen Eintrag hinzu
            aggregated_dict[poly_label] += " " + text
        else:
            # Erstelle einen neuen Eintrag im Dictionary für ein neues GT-Label
            aggregated_dict[poly_label] = text
    
    # Konvertiere das Dictionary zurück in eine Liste von Tupeln
    aggregated_data = [(label, text) for label, text in aggregated_dict.items()]
    return aggregated_data

def sum_sentences(sorted_data, i):
    sum_data = []
    current_first_part_of_polygon = None
    collected_text = ""

    # Überprüfe jedes Element in sorted_data für die korrekte Struktur
    for index, item in enumerate(sorted_data):
        if not isinstance(item, list) or len(item) < 1:
            print(f"Fehler bei Index {index}: Element hat nicht die erwartete Struktur {item}")
            continue  # Überspringe dieses Element

        # Gehe jedes Unter-Tupel in der Liste durch
        for sub_item in item:
            if not isinstance(sub_item, tuple) or len(sub_item) < 2 or not isinstance(sub_item[0], tuple):
                print(f"Fehler bei Index {index}: Unter-Element hat nicht die erwartete Struktur {sub_item}")
                continue  # Überspringe dieses Unter-Tupel

            # Zugriff auf das Polygon und den Text des Polygons
            polygon_with_text = sub_item[0]
            first_part_of_polygon, polygon_text = polygon_with_text[1], sub_item[2]

            if first_part_of_polygon == current_first_part_of_polygon:
                # Sicherstellen, dass nur Zeichenketten hinzugefügt werden
                if isinstance(polygon_text, str):
                    collected_text += " " + polygon_text
                else:
                    print("Warnung: Nicht-String-Wert in 'text' gefunden:", polygon_text)
            else:
                if current_first_part_of_polygon is not None:
                    # Sicherstellen, dass collected_text eine Zeichenkette ist
                    if isinstance(collected_text, str):
                        sum_data.append((current_first_part_of_polygon, collected_text.strip()))
                    else:
                        print("Warnung: 'collected_text' ist kein String:", collected_text)
                current_first_part_of_polygon = first_part_of_polygon
                collected_text = polygon_text if isinstance(polygon_text, str) else str(polygon_text)

    if current_first_part_of_polygon is not None and isinstance(collected_text, str):
        sum_data.append((current_first_part_of_polygon, collected_text.strip()))

    # Aggregiere Daten, um Duplikate zusammenzufassen
    final_sum_data = aggregate_sum_data(sum_data)

    csv_file_path = f'../results/sum_data_{i}.csv'
    df = pd.DataFrame(final_sum_data, columns=['GT Label', 'Predicted Text'])
    directory, filename = os.path.split(csv_file_path)
    os.makedirs(directory, exist_ok=True)
    df.to_csv(csv_file_path, index=False)

    save_sum_data_to_csv(final_sum_data, 'data_temp2.csv')

    return final_sum_data


def save_sum_data_to_csv(sum_data, output_file_path):
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['GT Label', 'Predicted Text'])
        for label, text in sum_data:
            writer.writerow([label, text])