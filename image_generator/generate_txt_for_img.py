import os
import random
import numpy as np
from tqdm import tqdm

def read_statistics(file_path):
    class_percentages = {}
    mean_objects = 0
    std_dev_objects = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Klasse" in line and "Mal gefunden" in line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[1][:-1])
                    percentage = float(parts[-3].strip('%').replace(',', '.'))
                    class_percentages[class_id] = percentage
            elif "Mittelwert" in line:
                mean_objects = float(line.split(':')[1].strip())
            elif "Standardabweichung" in line:
                std_dev_str = line.split(':')[1].strip()
                if std_dev_str:
                    std_dev_objects = float(std_dev_str)
                
    return class_percentages, mean_objects, std_dev_objects

def generate_class_distribution_file(class_percentages, num_lines, output_file, mean_objects, std_dev_objects):
    total_percentage = sum(class_percentages.values())
    lines = []

    for _ in tqdm(range(num_lines), desc="Generating class distribution"):
        line_distribution = {}
        num_objects = max(1, int(np.random.normal(mean_objects, std_dev_objects)))
        remaining_objects = num_objects
        
        # Berechnung der Grundverteilung basierend auf den Prozentsätzen
        for class_id, percentage in class_percentages.items():
            base_count = max(0, int(num_objects * (percentage / total_percentage)))
            line_distribution[class_id] = base_count
            remaining_objects -= base_count

        # Zufallsweise Verteilung der verbleibenden Objekte
        class_ids = list(class_percentages.keys())
        while remaining_objects > 0:
            class_id = random.choice(class_ids)
            line_distribution[class_id] += 1
            remaining_objects -= 1

        # Leichte Variation hinzufügen
        for class_id in line_distribution:
            if line_distribution[class_id] > 0:
                variation = random.randint(-3, 3)
                line_distribution[class_id] = max(0, line_distribution[class_id] + variation)

        lines.append(line_distribution)


    with open(output_file, 'w') as file:
        for line_distribution in lines:
            line = " ".join([f"Klasse {class_id}: {count}" for class_id, count in line_distribution.items()])
            file.write(line + "\n")

def verify_class_distribution(file_path, class_percentages, mean_objects, std_dev_objects, num_lines):
    total_lines = 0
    total_objects = 0
    cumulative_distribution = {class_id: 0 for class_id in class_percentages}
    all_objects_per_line = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            total_count = 0
            for i in range(0, len(parts), 3):
                class_id = int(parts[i+1][:-1])
                count = int(parts[i+2])
                total_count += count
                cumulative_distribution[class_id] += count

            total_lines += 1
            total_objects += total_count
            all_objects_per_line.append(total_count)

    avg_objects = total_objects / total_lines if total_lines > 0 else 0
    actual_std_dev_objects = np.std(all_objects_per_line) if total_lines > 0 else 0
    print("----------------------------------------------------------------------------------------------------")
    print(f"-- Verification completed. Average objects per line: {avg_objects:.2f} ± {actual_std_dev_objects:.2f} (Expected: {mean_objects} ± {std_dev_objects})")

    # Verify the overall distribution
    for class_id, expected_percentage in class_percentages.items():
        actual_percentage = (cumulative_distribution[class_id] / total_objects) * 100
        diff_percentage = actual_percentage - expected_percentage
        print(f"-- Class {class_id}: {actual_percentage:.2f}% (Expected: {expected_percentage:.2f}%) [Diff: {diff_percentage:.2f}%]")

    ''''
    # Calculate mean and standard deviation of positions
    for class_id, pos_list in positions.items():
        if pos_list:
            mean_x = np.mean([pos[0] for pos in pos_list])
            mean_y = np.mean([pos[1] for pos in pos_list])
            std_dev_x = np.std([pos[0] for pos in pos_list])
            std_dev_y = np.std([pos[1] for pos in pos_list])
            print(f"Klasse {class_id}:\n  Durchschnittliche Position: ({mean_x:.2f}, {mean_y:.2f})\n  Standardabweichung: ({std_dev_x:.2f}, {std_dev_y:.2f})")
    '''
            
