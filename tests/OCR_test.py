import cv2
import os
import numpy as np

def draw_bounding_boxes(image, texts, points_list, output_folder='test', base_name='test'):
    height, width = image.shape[:2]

    for text, points_str in zip(texts, points_list):
        points_float = [float(coord) for coord in points_str.split()]
        
        # Konvertieren normierter Koordinaten in Pixelkoordinaten
        points = np.array([(x * width, y * height) for x, y in zip(points_float[::2], points_float[1::2])], np.int32)
        
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path = os.path.join(output_folder, f"{base_name}_obb.jpg")
    cv2.imwrite(output_path, image)

image = cv2.imread('test.jpg')
text = ['Baum']
coordinates = ['0.7 0.6 0.7 0.3 0.9 0.3 0.9 0.6']
coordinates = ['0.0308150053024292 0.5003538727760315 0.03110022470355034 0.5704601407051086 0.9391499638557434 0.9695810317993164 0.13886475563049316 0.49947482347488403']



draw_bounding_boxes(image, text, coordinates)

print('done')