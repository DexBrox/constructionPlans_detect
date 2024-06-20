# main.py
import os
import glob
import cv2
from tqdm import tqdm
from itertools import product

from generate_txt_for_img import *
from image_generator_2 import *
from helper_functions_2 import *

# Einstellungen - Definiere hier deine gewünschten Parameter
rotation_ranges = [(0, 360)]
scale_ranges = [(0.5, 1.5)]
use_backgrounds_options = [False, True]
allow_overlap_options = [False, True]
name = 'synth_v3'

image_height = 2500
image_width = 3500
count_images = 1000

# Ordnerpfade
input_file = '/workspace/tests/statistic/stats_rp_v3_gesamt.txt'
output_file = 'class_distribution_rp_v3.txt'
backgrounds_folder = '/workspace/image_generator/background_rp_v3/*.png'
objects_folder = '/workspace/image_generator/objects_rp_v3/*.png'

# Txt erzeugen
class_percentages, mean_objects, std_dev_objects = read_statistics(input_file)
generate_class_distribution_file(class_percentages, count_images, output_file, mean_objects, std_dev_objects)

# Sicherstellen, dass die Ausgabeordner existieren
background_files = glob.glob(backgrounds_folder)
object_files = glob.glob(objects_folder)

# Klassenverteilung einlesen
class_distribution = read_class_distribution(output_file)

# Generiere alle Kombinationen der Einstellungen
settings_combinations = list(product(rotation_ranges, scale_ranges, use_backgrounds_options, allow_overlap_options))

# Datei, die alle Einstellungen speichert
all_settings_file = f'/workspace/datasets/synth/{name}_all_settings.txt'

# Durchlaufe jede Kombination der Einstellungen
for settings_index, (rotation_range, scale_range, use_backgrounds, allow_overlap) in enumerate(settings_combinations):
    if settings_index < 2:
        continue  # Überspringe die ersten beiden Kombinationen
    
    current_name = f"{name}_{settings_index+1}"
    output_folder_images = f'/workspace/datasets/synth/{current_name}/images/train'
    output_folder_labels = output_folder_images.replace('images', 'labels')

    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    # Bildgenerierung basierend auf den Klassenverteilungen
    for i, distribution in enumerate(tqdm(class_distribution, desc=f"Generating images for {current_name}")):
        generated_image, labels = place_objects_in_image_ft(
            background_files, object_files, image_height, image_width, distribution,
            rotation_range, scale_range, allow_overlap, use_backgrounds
        )

        output_file = os.path.join(output_folder_images, f"generated_image_{i+1}.png")
        label_file = os.path.join(output_folder_labels, f"generated_image_{i+1}.txt")

        cv2.imwrite(output_file, generated_image)

        with open(label_file, 'w') as lf:
            for label in labels:
                class_id, x1, y1, x2, y2, x3, y3, x4, y4 = label
                lf.write(f"{class_id} {x1/image_width:.4f} {y1/image_height:.4f} {x2/image_width:.4f} {y2/image_height:.4f} {x3/image_width:.4f} {y3/image_height:.4f} {x4/image_width:.4f} {y4/image_height:.4f}\n")

    print(f"Bilder und Labels für {current_name} wurden erfolgreich erzeugt und gespeichert.")

    # Speichere die aktuellen Einstellungen in eine zentrale Datei
    with open(all_settings_file, 'a') as sf:
        sf.write(f"Settings {settings_index+1}:\n")
        sf.write(f"Rotation Range={rotation_range}, Scale Range={scale_range}, Use Backgrounds={use_backgrounds}, Allow Overlap={allow_overlap}\n")
        sf.write("\n")

print(f"Alle Bilder und Labels wurden erfolgreich erzeugt und die Einstellungen in '{all_settings_file}' gespeichert.")
