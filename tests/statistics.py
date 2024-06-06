import os
import glob
import numpy as np
from collections import defaultdict

def count_labels(label_folders):
    class_counts = defaultdict(int)
    object_counts_per_image = []
    class_positions = defaultdict(list)

    for folder in label_folders:
        for label_file in glob.glob(os.path.join(folder, '*.txt')):
            with open(label_file, 'r') as file:
                object_count = 0
                for line in file:
                    parts = line.split()
                    class_name = parts[0]
                    coordinates = list(map(float, parts[1:]))
                    
                    # Aktualisieren der Klassenanzahl
                    class_counts[class_name] += 1
                    object_count += 1

                    # Speichern der Positionen für die Berechnung des Durchschnitts und der Standardabweichung
                    x_center = sum(coordinates[0::2]) / 4
                    y_center = sum(coordinates[1::2]) / 4
                    class_positions[class_name].append((x_center, y_center))
                
                object_counts_per_image.append(object_count)

    return class_counts, object_counts_per_image, class_positions

def format_output(class_counts, object_counts_per_image, class_positions):
    total_objects = sum(class_counts.values())
    sorted_counts = sorted(class_counts.items(), key=lambda item: int(item[0]))  # Sortiert nach Klassennummer
    class_percentages = [(class_name, count, (count / total_objects) * 100) for class_name, count in sorted_counts]
    
    # Berechnung der Statistik der Objekte pro Bild
    object_counts_array = np.array(object_counts_per_image)
    object_stats = {
        'mean': np.mean(object_counts_array),
        'median': np.median(object_counts_array),
        'std': np.std(object_counts_array)
    }
    
    # Berechnung der durchschnittlichen Positionen und Standardabweichungen
    class_position_stats = {}
    for class_name, positions in class_positions.items():
        positions_array = np.array(positions)
        mean_position = np.mean(positions_array, axis=0)
        std_position = np.std(positions_array, axis=0)
        class_position_stats[class_name] = {
            'mean_position': mean_position,
            'std_position': std_position
        }

    return class_percentages, object_stats, class_position_stats

def save_results(output_file, class_percentages, object_stats, class_position_stats):
    with open(output_file, 'w') as file:
        file.write("Verteilung der Klassen:\n")
        for class_name, count, percent in class_percentages:
            file.write(f"Klasse {class_name}: {count} Mal gefunden, {percent:.2f}% der Gesamtheit\n")
        
        file.write("\nObjekte pro Bild Statistik:\n")
        file.write(f"Mittelwert: {object_stats['mean']:.2f}\n")
        file.write(f"Median: {object_stats['median']:.2f}\n")
        file.write(f"Standardabweichung: {object_stats['std']:.2f}\n")
        
        file.write("\nDurchschnittliche Positionen und Standardabweichungen pro Klasse:\n")
        for class_name, stats in sorted(class_position_stats.items(), key=lambda item: int(item[0])):
            mean_x, mean_y = stats['mean_position']
            std_x, std_y = stats['std_position']
            file.write(f"Klasse {class_name}:\n")
            file.write(f"  Durchschnittliche Position: ({mean_x:.2f}, {mean_y:.2f})\n")
            file.write(f"  Standardabweichung: ({std_x:.2f}, {std_y:.2f})\n")

# Pfade zu den Verzeichnissen, wo die Label-Dateien gespeichert sind
label_folder1 = '/workspace/datasets/test2/synth_v1/labels/train'
#label_folder2 = label_folder1.replace('val', 'train')

#label_folder1 = '/workspace/datasets/standard/Roewaplan_org/labels/train'
#label_folder2 = label_folder1.replace('val', 'train')

# Zähle die Labels und sammle die Statistiken in den angegebenen Ordnern
class_counts, object_counts_per_image, class_positions = count_labels([label_folder1])#, label_folder2])

# Formatierung und Ausgabe der Ergebnisse
class_percentages, object_stats, class_position_stats = format_output(class_counts, object_counts_per_image, class_positions)

# Ergebnisse in einer Datei speichern
output_file = '/workspace/tests/statistic'
if not os.path.exists(output_file):
    os.makedirs(output_file)
save_results(f'{output_file}/analysis_results_synth_test2.txt', class_percentages, object_stats, class_position_stats)

print(f"Analyse abgeschlossen und Ergebnisse in '{output_file}' gespeichert.")
