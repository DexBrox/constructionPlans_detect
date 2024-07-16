import timeit
import os
from Levenshtein import distance as lv_distance
from OCR_evaluate import *


def calculate_cer(sum_data):
    total_errors = 0 

    #print(sum_data)

    total_chars = sum(len(gt) for gt, _ in sum_data)

    for gt, pred in sum_data:
        dist = lv_distance(gt, pred)

        total_errors += dist

    cer = total_errors / total_chars if total_chars > 0 else 0
    return cer


def evaluate_cer(gt_path, pred_file_path, i):
    
    gt, pred = load_files(gt_path, pred_file_path)
    gt_poly, poly_only_text = calculate_polygon(gt)
    pred_mid, pred_mid_w_t = calculate_midpoint(pred)
    linked_data = link_polygons_to_midpoints(gt_poly, poly_only_text, pred_mid, pred_mid_w_t)
    sorted_data = sort_linked_data_by_polygon_and_midpoint_x(linked_data)
    sum_data = sum_sentences(sorted_data, i)

    cer_results = calculate_cer(sum_data)

    return cer_results

def evaluate_cer_all_txt(input_dir_gt, output_dir_txt):
    txt_files = [f for f in os.listdir(output_dir_txt) if f.lower().endswith('.txt')]

    for txt_file in txt_files:
        file_id = os.path.splitext(txt_file)[0]  # Dateiname ohne Erweiterung
        gt_path = os.path.join(input_dir_gt, f'{file_id}.txt')
        pred_file_path = os.path.join(output_dir_txt, txt_file)

        if not os.path.exists(gt_path):
            print(f"Ground truth file {gt_path} not found.")
            continue
        
        start_eva = timeit.default_timer()
        cer_result = evaluate_cer(gt_path, pred_file_path, file_id)
        end_eva = timeit.default_timer()
        print(f"CER für {file_id}: {cer_result}")
        #print(f"Evaluationszeit für {file_id}: {end_eva - start_eva} Sekunden")
