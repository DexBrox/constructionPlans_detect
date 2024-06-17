import os
import glob
import cv2
from tqdm import tqdm
from helper_functions_2 import *

# Helferfunktionen definieren
def read_class_distribution(file_path):
    distributions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_counts = [int(count.split(': ')[1]) for count in line.split('Klasse ')[1:]]
            distributions.append(class_counts)
    return distributions

# Einstellungen - Definiere hier deine gewünschten Parameter
rotation_range = (0, 360)
scale_range = (0.5, 1.5)
use_backgrounds = True
allow_overlap = True
name = 'synth_v_custom'

image_height = 2500
image_width = 3500

# Ordnerpfade
backgrounds_folder = '/workspace/image_generator/backgrounds_rp_v2/*.jpg'
objects_folder = '/workspace/image_generator/objects_rp_v2/*.png'

# Sicherstellen, dass die Ausgabeordner existieren
background_files = glob.glob(backgrounds_folder)
object_files = glob.glob(objects_folder)

# Klassenverteilung einlesen
distribution_file = 'class_distribution_rp_v2.txt'
class_distribution = read_class_distribution(distribution_file)

output_folder_images = f'/workspace/datasets/synth_ft/{name}/images/train'
output_folder_labels = output_folder_images.replace('images', 'labels')

os.makedirs(output_folder_images, exist_ok=True)
os.makedirs(output_folder_labels, exist_ok=True)

# Bildgenerierung basierend auf den Klassenverteilungen
for i, distribution in tqdm(enumerate(class_distribution, desc=f"Generating images for {name}")):
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

print(f"Bilder und Labels für {name} wurden erfolgreich erzeugt und gespeichert.")

# Einstellungen speichern
settings_file = '/workspace/datasets/synthetisch_ft/settings.txt'
with open(settings_file, 'w') as sf:
    sf.write(f"Rotation={rotation_range}, Scale={scale_range}, Background={use_backgrounds}, Overlap={allow_overlap}\n")

print(f"Einstellungen wurden in '{settings_file}' gespeichert.")
