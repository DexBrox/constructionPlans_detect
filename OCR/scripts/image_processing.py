import cv2
import os
import numpy as np

def gen_out(image_path, image):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if base_name.endswith(".pdf"):
        base_name = base_name[:-4]
    return base_name


def draw_bounding_boxes(image, texts, points_list, output_folder, base_name):
    height, width = image.shape[:2]

    for text, points_str in zip(texts, points_list):
        points_float = [float(coord) for coord in points_str.split()]
        
        # Konvertieren normierter Koordinaten in Pixelkoordinaten
        points = np.array([(x * width, y * height) for x, y in zip(points_float[::2], points_float[1::2])], np.int32)
        
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path = os.path.join(output_folder, f"{base_name}_obb.jpg")
    cv2.imwrite(output_path, image)


def save_results(results, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{base_name}.txt")
    
    text_all = []
    coordinates_all = []

    with open(filename, 'w') as file:
        for result in results:
            parts = result.split()
            coordinates = ' '.join(parts[-8:])
            text = ' '.join(parts[:-8])

            file.write(f"{text} {coordinates}\n")

            text_all.append(text)
            coordinates_all.append(coordinates)

    return text_all, coordinates_all
