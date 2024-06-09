import os
import random
import numpy as np

def read_statistics(file_path):
    class_percentages = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Klasse" in line and "Mal gefunden" in line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[1][:-1])
                    percentage = float(parts[-3].strip('%').replace(',', '.'))
                    class_percentages[class_id] = percentage
    return class_percentages

def generate_class_distribution_file(class_percentages, num_lines, output_file, mean, std_dev):
    total_percentage = sum(class_percentages.values())
    lines = []
    for _ in range(num_lines):
        line_distribution = {}
        num_objects = max(1, int(np.random.normal(mean, std_dev)))
        remaining_objects = num_objects
        percentages = list(class_percentages.values())
        random.shuffle(percentages)  # Shuffle percentages to add randomness
        for class_id, percentage in class_percentages.items():
            if remaining_objects <= 0:
                line_distribution[class_id] = 0
            else:
                allocation = max(1, int(num_objects * (percentage / total_percentage)))
                allocation = min(allocation, remaining_objects)
                line_distribution[class_id] = allocation
                remaining_objects -= allocation
        lines.append(line_distribution)

    with open(output_file, 'w') as file:
        for line_distribution in lines:
            line = " ".join([f"Klasse {class_id}: {count}" for class_id, count in line_distribution.items()])
            file.write(line + "\n")

def verify_class_distribution(file_path, class_percentages, mean, std_dev):
    total_lines = 0
    total_objects = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_distribution = {}
            parts = line.split()
            total_count = 0
            for i in range(0, len(parts), 3):
                class_id = int(parts[i+1][:-1])
                count = int(parts[i+2])
                line_distribution[class_id] = count
                total_count += count

            total_percentage = sum(line_distribution.values())
            if total_percentage != total_count:
                print(f"Warning: Line total count does not match expected count: {line}")
            for class_id, expected_percentage in class_percentages.items():
                expected_count = int(mean * (expected_percentage / 100))
                actual_count = line_distribution.get(class_id, 0)
                if abs(actual_count - expected_count) > 1:
                    print(f"Warning: Class {class_id} does not match expected count: {actual_count} != {expected_count}")
            total_lines += 1
            total_objects += total_count

    avg_objects = total_objects / total_lines if total_lines > 0 else 0
    print(f"Verification completed. Average objects per line: {avg_objects:.2f} (Expected: {mean} ± {std_dev})")

# Beispielaufruf der Funktion
input_file = '/workspace/tests/statistic/analysis_results.txt'
output_file = 'class_distribution.txt'
num_lines = 10000
mean_objects = 39
std_dev_objects = 12

class_percentages = read_statistics(input_file)
generate_class_distribution_file(class_percentages, num_lines, output_file, mean_objects, std_dev_objects)
print(f"Class distribution file generated: {output_file}")
verify_class_distribution(output_file, class_percentages, mean_objects, std_dev_objects)
