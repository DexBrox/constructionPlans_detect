import os
import glob
import numpy as np

def is_rotated_box(coordinates, tolerance=0.00):
    if len(coordinates) != 8:
        return False
    
    def within_tolerance(a, b, tolerance):
        return abs(a - b) <= tolerance * abs(a)
    
    return not (
        within_tolerance(coordinates[0], coordinates[6], tolerance) and
        within_tolerance(coordinates[1], coordinates[3], tolerance) and
        within_tolerance(coordinates[2], coordinates[4], tolerance) and
        within_tolerance(coordinates[5], coordinates[7], tolerance)
    )

def count_rotated_boxes(label_folders):
    rotated_boxes_percentage = {}
    total_rotated_boxes_count = 0
    total_boxes_count = 0
    
    for folder in label_folders:
        rotated_boxes_count = 0
        folder_total_boxes_count = 0
        for label_file in glob.glob(os.path.join(folder, '*.txt')):
            with open(label_file, 'r') as file:
                for line in file:
                    parts = line.split()
                    coordinates = list(map(float, parts[1:]))
                    folder_total_boxes_count += 1
                    total_boxes_count += 1
                    if is_rotated_box(coordinates):
                        rotated_boxes_count += 1
                        total_rotated_boxes_count += 1
        rotated_boxes_percentage[folder] = (rotated_boxes_count / folder_total_boxes_count) * 100 if folder_total_boxes_count > 0 else 0
    
    total_percentage = (total_rotated_boxes_count / total_boxes_count) * 100 if total_boxes_count > 0 else 0
    
    return rotated_boxes_percentage, total_percentage

label_folder1 = '/workspace/datasets/standard/Roewaplan_v3/labels/train'
label_folder2 = label_folder1.replace('train', 'val')
label_folder3 = label_folder1.replace('train', 'test')

rotated_boxes_percentage, total_percentage = count_rotated_boxes([label_folder1, label_folder2, label_folder3])

print(f"Prozentuale Verteilung der gedrehten Bounding Boxen:")
for folder, percentage in rotated_boxes_percentage.items():
    folder_name = os.path.basename(folder)
    print(f"{folder_name}: {percentage:.2f}%")
    
print(f"Gesamtprozentuale Verteilung der gedrehten Bounding Boxen: {total_percentage:.2f}%")
