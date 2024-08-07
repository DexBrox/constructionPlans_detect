import os
import glob
import cv2
from tqdm import tqdm
import shutil as sh
import itertools
from image_generator.aa_outdated.helper_functions import place_objects_in_image, calculate_class_percentages

# Einstellungen
name_base = 'synth_v'
rotation_values = [(0, 0), (0, 360)]
scale_values = [(0.5, 1.5), (1.0, 1.0)]
background_values = [True, False]
overlap_values = [True, False]
image_height = 2500
image_width = 3500
num_objects = 45
num_generated_images = 100

# Ordnerpfade
backgrounds_folder = '/workspace/image_generator/backgrounds/*.jpg'
objects_folder = '/workspace/image_generator/objekte_white/*.png'

# Sicherstellen, dass die Ausgabeordner existieren
background_files = glob.glob(backgrounds_folder)
object_files = glob.glob(objects_folder)

# Klassenspezifische Verteilung
excluded_classes = {5, 8, 10, 11, 14}
class_percentages = calculate_class_percentages(object_files, excluded_classes, num_objects)

# Erstellen aller möglichen Kombinationen der Parameter
parameter_combinations = list(itertools.product(rotation_values, scale_values, background_values, overlap_values))

# Liste zum Speichern der Einstellungen
settings_list = []

for idx, combo in tqdm(enumerate(parameter_combinations, start=1)):
    rotation_range, scale_range, use_backgrounds, allow_overlap = combo
    name = f'{name_base}{idx}'
    
    output_folder_images = f'/workspace/datasets/test/synthetisch/{name}/images/train'
    output_folder_labels = output_folder_images.replace('images', 'labels')

    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    for i in range(num_generated_images):
        generated_image, labels = place_objects_in_image(background_files, object_files, image_height, image_width, num_objects, class_percentages, rotation_range, scale_range, allow_overlap, use_backgrounds)
        output_file = os.path.join(output_folder_images, f"generated_image_{i+1}.png")
        label_file = os.path.join(output_folder_labels, f"generated_image_{i+1}.txt")

        cv2.imwrite(output_file, generated_image)

        with open(label_file, 'w') as lf:
            for label in labels:
                class_id, x1, y1, x2, y2, x3, y3, x4, y4 = label
                lf.write(f"{class_id} {x1/image_width:.4f} {y1/image_height:.4f} {x2/image_width:.4f} {y2/image_height:.4f} {x3/image_width:.4f} {y3/image_height:.4f} {x4/image_width:.4f} {y4/image_height:.4f}\n")

    print(f"Bilder und Labels für {name} wurden erfolgreich erzeugt und gespeichert.")

    if not os.path.exists(output_folder_images.replace('train', 'val')):
        sh.copytree('/workspace/datasets/synthetisch/synth1_100_21/images/val', output_folder_images.replace('train', 'val'))
        print("Val-Bilder kopiert.")
        sh.copytree('/workspace/datasets/synthetisch/synth1_100_21/labels/val', output_folder_labels.replace('train', 'val'))
        print("Val-Labels kopiert.")
    
    # Einstellungen zur Liste hinzufügen
    settings_list.append(f"{idx}, Rotation={rotation_range}, Scale={scale_range}, Background={use_backgrounds}, Overlap={allow_overlap}")

# Einstellungen in einer Datei speichern
settings_file = '/workspace/datasets/synthetisch/settings.txt'
with open(settings_file, 'w') as sf:
    for setting in settings_list:
        sf.write(setting + '\n')

print(f"Alle {len(parameter_combinations)} Kombinationen wurden verarbeitet und in '{settings_file}' gespeichert.")
