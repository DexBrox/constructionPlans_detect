import os
import glob

def count_labels(label_folders):
    class_counts = {}

    for folder in label_folders:
        # Zugriff auf alle Label-Dateien im Verzeichnis
        for label_file in glob.glob(os.path.join(folder, '*.txt')):
            with open(label_file, 'r') as file:
                for line in file:
                    class_name = line.split()[0]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1

    return class_counts

def format_output(class_counts):
    total = sum(class_counts.values())
    sorted_counts = sorted(class_counts.items(), key=lambda item: int(item[0]))  # Sortiert nach Klassennummer
    percentages = [(class_name, count, (count / total) * 100) for class_name, count in sorted_counts]

    return percentages

# Pfade zu den Verzeichnissen, wo die Label-Dateien gespeichert sind
label_folder1 = '/workspace/datasets/Roewaplan/labels/val'
label_folder2 = '/workspace/datasets/Roewaplan/labels/train'

# Zähle die Labels in den angegebenen Ordnern
class_counts = count_labels([label_folder1, label_folder2])

# Formatierung und Ausgabe der Ergebnisse
formatted_output = format_output(class_counts)
for class_name, count, percent in formatted_output:
    print(f"Klasse {class_name}: {count} Mal gefunden, {percent:.2f}% der Gesamtheit")
