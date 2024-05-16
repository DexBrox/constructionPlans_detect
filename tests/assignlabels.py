import os
import shutil

def distribute_labels(label_dir, img_dir_train, img_dir_val, labels_dir_train, labels_dir_val):
    """
    Dieses Skript ordnet Label-Dateien aus einem zentralen Ordner den entsprechenden Bildern
    in den Trainings- und Validierungssets zu. Es wird überprüft, welche Bilder in den Ordnern
    'train/img' und 'val/img' vorhanden sind, und die zugehörigen Label-Dateien werden in die
    entsprechenden Ordner 'train/labels' und 'val/labels' verschoben.
    """
    # Erstelle die Zielordner, falls diese noch nicht existieren
    os.makedirs(labels_dir_train, exist_ok=True)
    os.makedirs(labels_dir_val, exist_ok=True)

    # Erstelle Sets der Bildnamen in den Trainings- und Validierungsordnern
    train_images = {f.split('.')[0] for f in os.listdir(img_dir_train)}
    val_images = {f.split('.')[0] for f in os.listdir(img_dir_val)}

    # Gehe durch alle Dateien im Label-Ordner
    for label_file in os.listdir(label_dir):
        base_name = label_file.split('.')[0]
        
        if base_name in train_images:
            # Verschiebe die Label-Datei in den Trainingsordner
            shutil.move(os.path.join(label_dir, label_file), os.path.join(labels_dir_train, label_file))
        elif base_name in val_images:
            # Verschiebe die Label-Datei in den Validierungsordner
            shutil.move(os.path.join(label_dir, label_file), os.path.join(labels_dir_val, label_file))

# Pfade anpassen
label_dir = 'label'
img_dir_train = '/workspace/datasets/Roewaplan/images/train'
img_dir_val = '/workspace/datasets/Roewaplan/images/val'
labels_dir_train = '/workspace/datasets/Roewaplan/labels/train'
labels_dir_val = '/workspace/datasets/Roewaplan/labels/val'

distribute_labels(label_dir, img_dir_train, img_dir_val, labels_dir_train, labels_dir_val)
