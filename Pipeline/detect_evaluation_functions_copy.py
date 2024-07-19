import glob
import os

def process_images(image, model):
    results = model.val([image])
    print("Evaluationsergebnisse:", results)
    return results


def save_results(results, image, img_folder, txt_folder):
    base_filename = os.path.basename(image)
    name, ext = os.path.splitext(base_filename)
    new_filename = f"{name}_d{ext}"
    new_filename_txt = f"{name}.txt"
    img_path_full = os.path.join(img_folder, new_filename)
    txt_path_full = os.path.join(txt_folder, new_filename_txt)

    for result in results:
        result.save(img_path_full, font_size=25)
        result.save_txt(txt_path_full)

    # Liste erzeugen mit Klasse und Koordinaten
    result_cls = result.obb.cls
    result_txt = result.obb.xyxyxyxyn

    data_list = []

    for cls, txt in zip(result_cls.tolist(), result_txt):
        class_name = int(cls) 
        coordinates = txt.squeeze(0).tolist()
        row = [class_name] + coordinates
        data_list.append(row)

    return data_list


def filter_text_files(data_list, target_class=7):
    data_list_filtered = [row for row in data_list if row[0] == target_class]

    return data_list_filtered

