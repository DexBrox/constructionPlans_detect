from PIL import Image

def crop_image(input_path, output_path, crop_coords):
    """
    Schneidet einen bestimmten Bereich aus einem Bild heraus und speichert das Ergebnis.

    :param input_path: Pfad zum Quellbild.
    :param output_path: Pfad, wo das zugeschnittene Bild gespeichert werden soll.
    :param crop_coords: Tuple von Koordinaten (x1, y1, x2, y2) für den Ausschnitt.
    """
    try:
        # Bild laden
        with Image.open(input_path) as img:
            # Bild zuschneiden
            cropped_image = img.crop(crop_coords)
            # Zugeschnittenes Bild speichern
            cropped_image.save(output_path)
            print(f"Bild erfolgreich zugeschnitten und gespeichert: {output_path}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

# Pfad zum Quellbild
input_image_path = '/workspace/OCR/results_rpv3/img/tesseract1/ALPHATON_Gen-06_QF_Sockel_obb.jpg'
#input_image_path = '/workspace/datasets/standard/Roewaplan_v3_visualized/test/ALPHATON_Gen-06_QF_Sockel_obb.png'

# Pfad, wo das zugeschnittene Bild gespeichert werden soll
output_image_path = '/workspace/OCR/zuschnitt/tesseract1.jpg'

# Koordinaten für den Zuschnitt (x1, y1, x2, y2)
coordinates = (100, 200, 1600, 1200)  # Beispielkoordinaten anpassen links, oben, rechts, unten

# Funktion zum Zuschneiden des Bildes aufrufen
crop_image(input_image_path, output_image_path, coordinates)
