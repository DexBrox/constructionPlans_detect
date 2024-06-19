import os
import glob
from PIL import Image
import numpy as np

def get_image_sizes(image_folders):
    sizes = []
    for image_folder in image_folders:
        for image_file in glob.glob(os.path.join(image_folder, '*.*')):
            try:
                with Image.open(image_file) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Error reading {image_file}: {e}")
    return sizes

def calculate_size_statistics(sizes):
    widths, heights = zip(*sizes)
    width_stats = {
        'mean': np.mean(widths),
        'median': np.median(widths),
        'std': np.std(widths),
        'min': np.min(widths),
        'max': np.max(widths)
    }
    height_stats = {
        'mean': np.mean(heights),
        'median': np.median(heights),
        'std': np.std(heights),
        'min': np.min(heights),
        'max': np.max(heights)
    }
    return width_stats, height_stats

def save_size_statistics(output_file, width_stats, height_stats):
    with open(output_file, 'w') as file:
        file.write("Bildgröße Statistik:\n")
        file.write(f"Weite Mittelwert: {width_stats['mean']:.2f}\n")
        file.write(f"Weite Median: {width_stats['median']:.2f}\n")
        file.write(f"Weite Standardabweichung: {width_stats['std']:.2f}\n")
        file.write(f"Weite Kleinste: {width_stats['min']}\n")
        file.write(f"Weite Größte: {width_stats['max']}\n")
        file.write(f"Höhe Mittelwert: {height_stats['mean']:.2f}\n")
        file.write(f"Höhe Median: {height_stats['median']:.2f}\n")
        file.write(f"Höhe Standardabweichung: {height_stats['std']:.2f}\n")
        file.write(f"Höhe Kleinste: {height_stats['min']}\n")
        file.write(f"Höhe Größte: {height_stats['max']}\n")

if __name__ == "__main__":
    img_folder1 = '/workspace/datasets/standard/Roewaplan_v2/images/train'
    img_folder2 = img_folder1.replace('train', 'val')
    img_folder3 = img_folder1.replace('train', 'test')
    
    image_sizes = get_image_sizes([img_folder1, img_folder2, img_folder3])
    width_stats, height_stats = calculate_size_statistics(image_sizes)

    output_file = '/workspace/tests/statistic/analysis_results_image_sizes.txt'
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    save_size_statistics(output_file, width_stats, height_stats)

    print(f"Analyse abgeschlossen und Ergebnisse in '{output_file}' gespeichert.")
