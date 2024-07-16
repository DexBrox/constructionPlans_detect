import os
import timeit
import warnings

start = timeit.default_timer()

from OCR_text_recognition_eo import process_image_easyocr
from OCR_text_recognition_tess import process_image_tess
from OCR_text_recognition_hybrid import process_image_hy
from OCR_image_processing import draw_bounding_boxes, gen_out, save_results

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Setzen der Pfade
input_dir_img = '/workspace/datasets/standard/Roewaplan_v3/images/test'  # Direktordner mit Bildern
input_dir_gt = input_dir_img.replace('images', 'labels')  # GT-Ordner

output_dir_img = '../results_rpv3_test/img/tesseract1'  
output_dir_txt = '../results_rpv3_test/txt'

# Bilder im Verzeichnis auflisten
image_paths = [os.path.join(input_dir_img, file) for file in os.listdir(input_dir_img) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

count = 0
time_per_image = []

for image_path in image_paths:
    image_start = timeit.default_timer()  # Startzeit für die Verarbeitung des einzelnen Bildes

    #results, image = process_image_easyocr(image_path)
    results, image = process_image_tess(image_path)
    base_name = gen_out(image_path, image)
    text, coordinates = save_results(results, output_dir_txt, base_name)
    draw_bounding_boxes(image, text, coordinates, output_dir_img, base_name)

    image_end = timeit.default_timer()  # Endzeit für die Verarbeitung des einzelnen Bildes
    processing_time = image_end - image_start
    time_per_image.append(processing_time)  # Speichern der Zeit für jedes Bild

    print(f"Verarbeitungszeit für {os.path.basename(image_path)}: {processing_time} Sekunden")

    count += 1

end = timeit.default_timer()
if count > 0:
    print(f"Durchschnittliche Verarbeitungszeit: {(end - start) / count} Sekunden pro Bild")

from OCR_cer import evaluate_cer_all_txt

evaluate_cer_all_txt(input_dir_gt, output_dir_txt)
