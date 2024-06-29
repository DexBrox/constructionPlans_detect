import os
import glob
import numpy as np
from collections import defaultdict

def count_labels(label_folders):
    class_counts = defaultdict(int)
    object_counts_per_image = []
    class_positions = defaultdict(list)
    total_images = 0

    for folder in label_folders:
        for label_file in glob.glob(os.path.join(folder, '*.txt')):
            print(f"Analysiere {label_file}")
            with open(label_file, 'r') as file:
                object_count = 0
                for line in file:
                    parts = line.split()
                    class_name = parts[0]
                    coordinates = list(map(float, parts[1:]))
                    
                    class_counts[class_name] += 1
                    object_count += 1

                    x_center = sum(coordinates[0::2]) / 4
                    y_center = sum(coordinates[1::2]) / 4
                    class_positions[class_name].append((x_center, y_center))
                
                object_counts_per_image.append(object_count)
            total_images += 1

    return class_counts, object_counts_per_image, class_positions, total_images

def format_output(class_counts, object_counts_per_image, class_positions, total_images):
    total_objects = sum(class_counts.values())
    sorted_counts = sorted(class_counts.items(), key=lambda item: int(item[0]))
    class_percentages = [(class_name, count, (count / total_objects) * 100) for class_name, count in sorted_counts]
    
    object_counts_array = np.array(object_counts_per_image)
    object_stats = {
        'mean': np.mean(object_counts_array),
        'median': np.median(object_counts_array),
        'std': np.std(object_counts_array),
        'total_images': total_images
    }
    
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
        file.write(f"Anzahl der Bilder: {object_stats['total_images']}\n")
        
        file.write("\nDurchschnittliche Positionen und Standardabweichungen pro Klasse:\n")
        for class_name, stats in sorted(class_position_stats.items(), key=lambda item: int(item[0])):
            mean_x, mean_y = stats['mean_position']
            std_x, std_y = stats['std_position']
            file.write(f"Klasse {class_name}:\n")
            file.write(f"  Durchschnittliche Position: ({mean_x:.2f}, {mean_y:.2f})\n")
            file.write(f"  Klasse Stabw: ({std_x:.2f}, {std_y:.2f})\n")

label_folder1 = '/workspace/datasets/synth/synth_v3_200_1/labels/train'
#label_folder2 = label_folder1.replace('train', 'val')
#label_folder3 = label_folder1.replace('train', 'test')

class_counts, object_counts_per_image, class_positions, total_images = count_labels([label_folder1])#, label_folder2, label_folder3])

class_percentages, object_stats, class_position_stats = format_output(class_counts, object_counts_per_image, class_positions, total_images)

output_path = '/workspace/tests/statistic'
if not os.path.exists(output_path):
    os.makedirs(output_path)
save_results(f'{output_path}/stats_synth_v3.txt', class_percentages, object_stats, class_position_stats)

print(f"Analyse abgeschlossen und Ergebnisse in '{output_path}' gespeichert.")
