import os
import shutil
import random

def split_dataset(base_folder, train_ratio, val_ratio, test_ratio):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Die Summe der Verhältnisse für Training, Validierung und Test muss 1.0 ergeben.")

    image_base_folder = os.path.join(base_folder, 'images', 'org')
    label_base_folder = os.path.join(base_folder, 'labels', 'org')

    train_image_folder = os.path.join(base_folder, 'images', 'train')
    val_image_folder = os.path.join(base_folder, 'images', 'val')
    test_image_folder = os.path.join(base_folder, 'images', 'test')
    train_label_folder = os.path.join(base_folder, 'labels', 'train')
    val_label_folder = os.path.join(base_folder, 'labels', 'val')
    test_label_folder = os.path.join(base_folder, 'labels', 'test')

    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    images = [file for file in os.listdir(image_base_folder) if file.endswith(('.jpg', '.png'))]

    random.shuffle(images)
    random.shuffle(images)
    random.shuffle(images)

    total_images = len(images)
    train_split_index = int(total_images * train_ratio)
    val_split_index = train_split_index + int(total_images * val_ratio)
    
    train_images = images[:train_split_index]
    val_images = images[train_split_index:val_split_index]
    test_images = images[val_split_index:]

    print(f"Anzahl der Trainingsbilder: {len(train_images)}")
    print(f"Anzahl der Validierungsbilder: {len(val_images)}")
    print(f"Anzahl der Testbilder: {len(test_images)}")

    new_train_count = int(input(f"Geben Sie die gewünschte Anzahl der Trainingsbilder ein (aktuell {len(train_images)}): "))
    new_val_count = int(input(f"Geben Sie die gewünschte Anzahl der Validierungsbilder ein (aktuell {len(val_images)}): "))
    new_test_count = int(input(f"Geben Sie die gewünschte Anzahl der Testbilder ein (aktuell {len(test_images)}): "))

    if new_train_count + new_val_count + new_test_count != total_images:
        raise ValueError("Die Summe der angegebenen Werte muss der Gesamtzahl der Bilder entsprechen.")

    train_images = images[:new_train_count]
    val_images = images[new_train_count:new_train_count + new_val_count]
    test_images = images[new_train_count + new_val_count:new_train_count + new_val_count + new_test_count]

    def move_files(files, destination_image_folder, destination_label_folder):
        for file in files:
            shutil.move(os.path.join(image_base_folder, file), os.path.join(destination_image_folder, file))
            label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_src_path = os.path.join(label_base_folder, label_file)
            if os.path.exists(label_src_path):
                label_dst_path = os.path.join(destination_label_folder, label_file)
                shutil.move(label_src_path, label_dst_path)

    move_files(train_images, train_image_folder, train_label_folder)
    move_files(val_images, val_image_folder, val_label_folder)
    move_files(test_images, test_image_folder, test_label_folder)

    shutil.rmtree(image_base_folder)
    shutil.rmtree(label_base_folder)

    print(f'Verschoben {len(train_images)} Bilder nach {train_image_folder}')
    print(f'Verschoben {len(val_images)} Bilder nach {val_image_folder}')
    print(f'Verschoben {len(test_images)} Bilder nach {test_image_folder}')
    print(f'Der Ordner {image_base_folder} und {label_base_folder} wurden gelöscht.')

if __name__ == "__main__":
    base_folder = '/workspace/datasets/standard/Roewaplan_v2'
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    split_dataset(base_folder, train_ratio, val_ratio, test_ratio)
