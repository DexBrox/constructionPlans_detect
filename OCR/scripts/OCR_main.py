import os
import timeit
import warnings

start = timeit.default_timer()

from OCR_pdf_to_image import convert_pdf_to_images
from OCR_text_recognition_eo import process_image_easyocr
from OCR_text_recognition_tess import process_image_tess
from OCR_text_recognition_hybrid import process_image_hy
from OCR_image_processing import draw_bounding_boxes, gen_out, save_results

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Setzten der Pfade
input_dir_img = '../pdf'
input_dir_gt = '../labels'

output_dir_img = '../pdf/img'
output_dir_txt = '../results/txt'


image_paths = convert_pdf_to_images(input_dir_img, output_dir_img)

count = 0
for image_path in image_paths:
    #results, image = process_image_easyocr(image_path)
    results, image = process_image_tess(image_path)
    base_name = gen_out(image_path, image)
    text, coordinates = save_results(results, output_dir_txt, base_name)

    draw_bounding_boxes(image, text, coordinates, output_dir_txt, base_name)
    count += 1

end = timeit.default_timer()
print(f"Verarbeitungszeit: {(end - start)/count} Sekunden pro Bild")

from cer import evaluate_cer

i = []
for i in range(1, 3):
    start_eva = timeit.default_timer()

    cer_result = evaluate_cer(input_dir_gt, output_dir_txt, i)
    print(f"CER für {i}: {cer_result}")
    end_eva = timeit.default_timer()

    print(f"Evaluationszeit für {i}: {end_eva - start_eva} Sekunden")



