import os
import numpy as np

def parse_distribution_file(file_path):
    class_counts = {}
    total_objects = 0
    num_lines = 0
    objects_per_line = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            line_total = 0
            for i in range(0, len(parts), 3):
                class_id = int(parts[i+1][:-1])
                count = int(parts[i+2])
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += count
                line_total += count
                total_objects += count
            objects_per_line.append(line_total)
            num_lines += 1

    return class_counts, total_objects, num_lines, objects_per_line

def calculate_statistics(class_counts, total_objects, num_lines, objects_per_line):
    class_statistics = {}
    for class_id, count in class_counts.items():
        percentage = (count / total_objects) * 100
        class_statistics[class_id] = (count, percentage)

    mean_objects_per_line = np.mean(objects_per_line)
    median_objects_per_line = np.median(objects_per_line)
    std_dev_objects_per_line = np.std(objects_per_line)

    return class_statistics, mean_objects_per_line, median_objects_per_line, std_dev_objects_per_line

def generate_statistics_report(file_path, output_file):
    class_counts, total_objects, num_lines, objects_per_line = parse_distribution_file(file_path)
    class_statistics, mean_objects_per_line, median_objects_per_line, std_dev_objects_per_line = calculate_statistics(class_counts, total_objects, num_lines, objects_per_line)

    with open(output_file, 'w') as file:
        file.write("Verteilung der Klassen:\n")
        for class_id, (count, percentage) in class_statistics.items():
            file.write(f"Klasse {class_id}: {count} Mal gefunden, {percentage:.2f}% der Gesamtheit\n")

        file.write("\nObjekte pro Bild Statistik:\n")
        file.write(f"Mittelwert: {mean_objects_per_line:.2f}\n")
        file.write(f"Median: {median_objects_per_line:.2f}\n")
        file.write(f"Standardabweichung: {std_dev_objects_per_line:.2f}\n")

# Beispielaufruf der Funktion
input_file = 'class_distribution_rp_v2.txt'
output_file = 'class_distribution_statistics.txt'

generate_statistics_report(input_file, output_file)
print(f"Statistics report generated: {output_file}")
