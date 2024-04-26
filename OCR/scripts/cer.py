import timeit
from Levenshtein import distance as lv_distance
from evaluate import *


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
    
    gt, pred = load_files(gt_path, pred_file_path, i)
    gt_poly, poly_only_text = calculate_polygon(gt)
    pred_mid, pred_mid_w_t = calculate_midpoint(pred)
    linked_data = link_polygons_to_midpoints(gt_poly, poly_only_text, pred_mid, pred_mid_w_t)
    sorted_data = sort_linked_data_by_polygon_and_midpoint_x(linked_data)
    sum_data = sum_sentences(sorted_data, i)

    cer_results = calculate_cer(sum_data)

    return cer_results
