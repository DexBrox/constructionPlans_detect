import os
import shutil
import random

def split_dataset(base_folder, train_ratio=0.7):
    # Basispfad für Bilder und Labels
    image_base_folder = os.path.join(base_folder, 'images', 'org')
    label_base_folder = os.path.join(base_folder, 'labels')

    # Pfade für Trainings- und Testdatensätze
    train_image_folder = os.path.join(base_folder, 'images', 'train')
    test_image_folder = os.path.join(base_folder, 'images', 'test')
    train_label_folder = os.path.join(label_base_folder, 'train')
    test_label_folder = os.path.join(label_base_folder, 'test')

    # Erstelle die Ordner, wenn sie nicht existieren
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    # Liste aller Bild-Dateien im Basisordner der Bilder
    images = [file for file in os.listdir(image_base_folder) if file.endswith(('.jpg', '.png'))]

    # Mische die Bilder zufällig
    random.shuffle(images)

    # Trenne Bilder in Trainings- und Testdatensätze
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Funktion, um Dateien zu verschieben
    def move_files(files, destination_image_folder, destination_label_folder):
        for file in files:
            # Bewege Bild
            shutil.move(os.path.join(image_base_folder, file), os.path.join(destination_image_folder, file))
            # Bewege zugehörige Label-Datei
            label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_src_path = os.path.join(image_base_folder, label_file)  # Annahme: Labels liegen zunächst im selben Ordner
            label_dst_path = os.path.join(destination_label_folder, label_file)
            shutil.move(label_src_path, label_dst_path)

    # Bewege Dateien in entsprechende Ordner
    move_files(train_images, train_image_folder, train_label_folder)
    move_files(test_images, test_image_folder, test_label_folder)

    print(f'Verschoben {len(train_images)} Bilder nach {train_image_folder}')
    print(f'Verschoben {len(test_images)} Bilder nach {test_image_folder}')

# Beispielaufruf
base_folder = '/workspace/datasets/Roewaplan_misch'  # Setze den Pfad zum Hauptordner
split_dataset(base_folder)
