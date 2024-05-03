import glob
import os

def process_images(image_folder, model, img_folder, txt_folder):
    image_files = glob.glob(f'{image_folder}/*.jpg')
    for image in image_files:
        results = model.predict([image], conf=0.15)
        save_results(results, image, img_folder, txt_folder)

def save_results(results, image_path, img_folder, txt_folder):
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    new_filename = f"{name}_d{ext}"
    new_filename_txt = f"{name}.txt"
    img_path_full = os.path.join(img_folder, new_filename)
    txt_path_full = os.path.join(txt_folder, new_filename_txt)

    for result in results:
        result.save(img_path_full, font_size=25)
        result.save_txt(txt_path_full)

def filter_text_files(txt_folder, target_folder):
    txt_files = glob.glob(f'{txt_folder}/*.txt')
    for file_path in txt_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if line.strip().split()[0] == '7']
        base_name = os.path.basename(file_path)
        new_file_path = os.path.join(target_folder, base_name)

        with open(new_file_path, 'w') as new_file:
            new_file.writelines(filtered_lines)
