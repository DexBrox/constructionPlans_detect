import cv2
import easyocr

def process_image_easyocr(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['de', 'en', 'fr'], gpu=True)
    results = reader.readtext(image, rotation_info=[0, 90, 180, 270], text_threshold=0.3)
    
    image_height, image_width = image.shape[:2]

    results_obb = []
    for (bbox, text, prob) in results:
        line = f"{text} {bbox[0][0]/image_width} {bbox[0][1]/image_height} {bbox[1][0]/image_width} {bbox[1][1]/image_height} {bbox[2][0]/image_width} {bbox[2][1]/image_height} {bbox[3][0]/image_width} {bbox[3][1]/image_height}"
        results_obb.append(line)

    return results_obb, image

