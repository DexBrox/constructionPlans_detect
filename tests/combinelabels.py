import os

def read_labels_from_file(file_path):
    """Liest Labels aus einer Datei und gibt sie als Liste zurück."""
    with open(file_path, 'r') as file:
        labels = file.readlines()
    return labels

def write_labels_to_file(labels, output_file_path):
    """Schreibt die kombinierten Labels in eine Datei."""
    with open(output_file_path, 'w') as file:
        file.writelines(labels)


def change_class(labels):
    for label in labels:
        #print first entry from labe
        label = label.split()
        print (label[0])
        #replace first entry wth 7
        label[0] = '7'
        print (label[0])
    print(labels)
    return labels

def combine_labels(input_folder1, input_folder2, output_folder):
    """Kombiniert Labels aus zwei Ordnern und speichert sie unter demselben Namen in einem neuen Ordner."""
    # Sicherstellen, dass der Ausgabeordner existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste aller Dateien in den Eingabeordnern
    files1 = [f for f in os.listdir(input_folder1) if os.path.isfile(os.path.join(input_folder1, f))]

    for file_name in files1:
        file_path1 = os.path.join(input_folder1, file_name)
        file_path2 = os.path.join(input_folder2, file_name)
        
        combined_labels = []
        
        # Lesen und Kombinieren der Labels, falls die Datei in beiden Ordnern existiert
        if os.path.exists(file_path2):
            labels1_f = read_labels_from_file(file_path1)
            labels1 = change_class(labels1_f)
            labels2 = read_labels_from_file(file_path2)
            combined_labels.extend(labels1)
            combined_labels.extend(labels2)
        elif os.path.exists(file_path1):
            combined_labels.extend(read_labels_from_file(file_path1))
        
        # Ausgabe-Pfad erstellen und die kombinierten Labels speichern
        output_file_path = os.path.join(output_folder, file_name)
        write_labels_to_file(combined_labels, output_file_path)

# Beispielverwendung
input_folder1 = '/workspace/cvat2yolo-conversion-main/labels_ocr'
input_folder2 = input_folder1.replace('labels_ocr', 'labels_od')
output_folder = input_folder1.replace('labels_ocr', 'labels_all')

combine_labels(input_folder1, input_folder2, output_folder)
