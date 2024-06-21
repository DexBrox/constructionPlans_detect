import xml.etree.ElementTree as ET
import os
import math

def rotate_point(cx, cy, angle_deg, px, py):
    """ Rotiere einen Punkt um einen anderen Punkt. Angle ist in Grad. """
    angle_rad = math.radians(angle_deg)  # Umwandlung von Grad in Bogenmaß
    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    # translate point back to origin:
    px -= cx
    py -= cy

    # rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # translate point back:
    px = xnew + cx
    py = ynew + cy
    return px, py

def extract_and_transform_data(xml_file_path, output_directory):
    """
    Dieses Skript extrahiert Objektannotationsdaten aus einer XML-Datei, die im CVAT-Format vorliegen.
    Für jede Bild-Annotation werden die Koordinaten der Bounding Boxes extrahiert, rotiert und
    auf die Größe des Bildes normiert. Diese Daten werden dann in textbasierten Label-Dateien gespeichert,
    wobei jede Datei den Annotationen eines Bildes entspricht.
    Die Labels werden ebenfalls von ihren Namen in nummerierte Indizes umgewandelt, die in einem Mapping festgelegt sind.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Label Mapping erstellen
    label_mapping = {}
    labels_element = root.find('.//labels')
    for i, label_element in enumerate(labels_element.findall('label')):
        label_name = label_element.find('name').text
        label_mapping[label_name] = i

    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for image in root.findall('.//image'):
        image_name = image.get('name').split('.')[0]  # Bildname ohne Dateiendung
        output_file_path = os.path.join(output_directory, f"{image_name}.txt")

        # Bildgröße extrahieren
        width = float(image.get('width'))
        height = float(image.get('height'))
        
        with open(output_file_path, 'w') as file:
            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                rotation = float(box.get('rotation')) if box.get('rotation') else 0.0  # Winkel in Grad

                label_index = label_mapping.get(label, -1)  # Zuweisung der Label-Indizes

                # Berechne den Mittelpunkt
                cx = (xtl + xbr) / 2
                cy = (ytl + ybr) / 2

                # Berechne rotierte Punkte
                x1, y1 = rotate_point(cx, cy, rotation, xtl, ytl)
                x2, y2 = rotate_point(cx, cy, rotation, xbr, ytl)
                x3, y3 = rotate_point(cx, cy, rotation, xbr, ybr)
                x4, y4 = rotate_point(cx, cy, rotation, xtl, ybr)

                # Normalisiere die Koordinaten auf die Bildgröße
                x1 /= width
                y1 /= height
                x2 /= width
                y2 /= height
                x3 /= width
                y3 /= height
                x4 /= width
                y4 /= height

                file.write(f"{label_index} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")

# Pfad zu deiner XML-Datei und Zielverzeichnis
input_xml_file_path = 'new_labels_13_06/annotations.xml'
output_directory = 'new_labels_13_06/labels'

extract_and_transform_data(input_xml_file_path, output_directory)
