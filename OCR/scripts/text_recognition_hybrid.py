import cv2
import pytesseract
import easyocr

def process_image_hy(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['de', 'en'], gpu=True)
    results = reader.readtext(image, rotation_info=[0, 90, 180, 270])

    for i, (bbox, text, confidence) in enumerate(results):
        offset = 0
        # Bounding Box Koordinaten extrahieren
        (top_left, top_right, bottom_right, bottom_left) = bbox
        # Bild basierend auf Bounding Box zuschneiden
        x, y = int(top_left[0]), int(top_left[1])
        w, h = int(top_right[0] - top_left[0]), int(bottom_left[1] - top_left[1])
        #print (x, y, w, h)
        cropped_image = image[max(0, y-offset):min(y+h+offset, image.shape[0]), max(0, x-offset):min(x+w+offset, image.shape[1])]
        
        # Text innerhalb der zugeschnittenen Bounding Box mit PyTesseract erkennen
        config = '--oem 3 --psm 1 -l deu+eng'
        pytesseract_text = pytesseract.image_to_string(cropped_image, config=config, lang='deu+eng')

        # Ergebnis mit dem neuen Text aktualisieren
        results[i] = (bbox, pytesseract_text.strip(), confidence)

    return results, image
