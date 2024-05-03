import cv2
import pytesseract

def process_image_tess(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Höhe und Breite des Bildes

    conf = 0.0

    config = '--oem 3 --psm 1 -l deu+eng'
    boxes = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

    results = []
    n_boxes = len(boxes['level'])
    for i in range(n_boxes):
        if boxes['text'][i].strip() != '' and float(boxes['conf'][i]) > conf*100:
            (x, y, box_w, box_h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            # Normiere die Koordinaten
            norm_coords = [x/w, y/h, (x + box_w)/w, y/h, (x + box_w)/w, (y + box_h)/h, x/w, (y + box_h)/h]
            # Erstelle den formatierten String
            result_str = f"{boxes['text'][i]} {' '.join(map(str, norm_coords))}"
            results.append(result_str)

    return results, image
