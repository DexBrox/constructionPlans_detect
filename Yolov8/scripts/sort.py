import os
import glob

source_folder = '/workspace/Yolov8/results/txt'
target_folder = '/workspace/Yolov8/results/txt_beschriftung'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for file_path in glob.glob(os.path.join(source_folder, '*.txt')):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if line.strip().split()[0] == '7']

    base_name = os.path.basename(file_path)
    new_file_path = os.path.join(target_folder, base_name)

    with open(new_file_path, 'w') as new_file:
        new_file.writelines(filtered_lines)