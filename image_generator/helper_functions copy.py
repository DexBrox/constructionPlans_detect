import os
import cv2
import numpy as np
import random

def do_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def place_objects_in_image(background_files, objects, image_height, image_width, num_objects, class_percentages, rotation_range, scale_range, allow_overlap, use_backgrounds):
    if use_backgrounds:
        background_path = random.choice(background_files)
        image = cv2.imread(background_path)
        if image is None:
            image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        else:
            image = cv2.resize(image, (image_width, image_height))
    else:
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    
    labels = []
    placed_objects = []
    class_counts = {class_id: int(num_objects * (percentage / 100)) for class_id, percentage in class_percentages.items()}

    for class_id, count in class_counts.items():
        for _ in range(count):
            obj_path = random.choice([obj for obj in objects if int(os.path.basename(obj).split('_')[0]) == class_id])
            obj_name = os.path.basename(obj_path)
            obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            
            if obj is None:
                continue
            
            obj_height, obj_width = obj.shape[:2]
            scale = random.uniform(scale_range[0], scale_range[1])
            rotation = random.randint(rotation_range[0], rotation_range[1])

            # Transformation Matrix
            M = cv2.getRotationMatrix2D((obj_width / 2, obj_height / 2), rotation, scale)

            # Calculate new width and height of the rotated object
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_width = int((obj_height * sin) + (obj_width * cos))
            new_height = int((obj_height * cos) + (obj_width * sin))

            # Adjust the transformation matrix to account for the new width and height
            M[0, 2] += (new_width / 2) - (obj_width / 2)
            M[1, 2] += (new_height / 2) - (obj_height / 2)

            # Apply the transformation to the object
            rotated_obj = cv2.warpAffine(obj, M, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            max_attempts = 100
            for attempt in range(max_attempts):
                x_offset = random.randint(0, image_width - new_width)
                y_offset = random.randint(0, image_height - new_height)

                overlap = False
                for (px, py, pw, ph, pc, _) in placed_objects:
                    if do_overlap(x_offset, y_offset, new_width, new_height, px, py, pw, ph):
                        overlap = True
                        break
                if overlap and not allow_overlap:
                    continue

                placed_objects.append((x_offset, y_offset, new_width, new_height, class_id, obj_name))
                break
            else:
                # Wenn keine geeignete Position gefunden wurde, brechen wir die Schleife ab
                continue

            for c in range(3):
                alpha = rotated_obj[:, :, 3] / 255.0
                image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] = \
                    rotated_obj[:, :, c] * alpha + image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] * (1.0 - alpha)

            # Calculate the four corners of the bounding box before rotation
            points = np.array([
                [0, 0],
                [obj_width, 0],
                [obj_width, obj_height],
                [0, obj_height]
            ], dtype=np.float32)

            # Apply the same transformation to the bounding box points
            transformed_points = cv2.transform(np.array([points]), M)[0]
            
            # Apply the offset
            transformed_points[:, 0] += x_offset
            transformed_points[:, 1] += y_offset

            x1, y1 = transformed_points[0]
            x2, y2 = transformed_points[1]
            x3, y3 = transformed_points[2]
            x4, y4 = transformed_points[3]

            labels.append((class_id, x1, y1, x2, y2, x3, y3, x4, y4))

    return image, labels

def place_objects_in_image_ft(background_files, objects, image_height, image_width, num_objects, num_objects_std, class_percentages, rotation_range, scale_range, allow_overlap, use_backgrounds):
    if use_backgrounds:
        background_path = random.choice(background_files)
        image = cv2.imread(background_path)
        if image is None:
            image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        else:
            image = cv2.resize(image, (image_width, image_height))
    else:
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    
    labels = []
    placed_objects = []
    
    total_percentage = sum(class_percentages.values())
    random_percentage = random.uniform(0.1, 5.0)
    target_count = int(np.random.normal(num_objects, num_objects_std))
    class_counts = {class_id: max(0, int(target_count * (percentage / total_percentage) * random_percentage)) for class_id, percentage in class_percentages.items()}
    class_counts = {class_id: int(round(count / random_percentage)) for class_id, count in class_counts.items()}
    print(class_counts)

    for class_id, count in class_counts.items():
        available_objects = [obj for obj in objects if int(os.path.basename(obj).split('_')[0]) == int(class_id)]
        if not available_objects:
            print(f"Warning: No objects found for class_id {class_id}")
            continue

        for _ in range(count):
            obj_path = random.choice(available_objects)
            obj_name = os.path.basename(obj_path)
            obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            
            if obj is None:
                print(f"Error: Unable to read object image {obj_path}")
                continue
            
            obj_height, obj_width = obj.shape[:2]
            scale = random.uniform(scale_range[0], scale_range[1])
            rotation = random.randint(rotation_range[0], rotation_range[1])

            # Transformation Matrix
            M = cv2.getRotationMatrix2D((obj_width / 2, obj_height / 2), rotation, scale)

            # Calculate new width and height of the rotated object
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_width = int((obj_height * sin) + (obj_width * cos))
            new_height = int((obj_height * cos) + (obj_width * sin))

            # Adjust the transformation matrix to account for the new width and height
            M[0, 2] += (new_width / 2) - (obj_width / 2)
            M[1, 2] += (new_height / 2) - (obj_height / 2)

            # Apply the transformation to the object
            rotated_obj = cv2.warpAffine(obj, M, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            max_attempts = 100
            for attempt in range(max_attempts):
                x_offset = random.randint(0, image_width - new_width)
                y_offset = random.randint(0, image_height - new_height)

                overlap = False
                for (px, py, pw, ph, pc, _) in placed_objects:
                    if do_overlap(x_offset, y_offset, new_width, new_height, px, py, pw, ph):
                        overlap = True
                        break
                if overlap and not allow_overlap:
                    continue

                placed_objects.append((x_offset, y_offset, new_width, new_height, class_id, obj_name))
                break
            else:
                # Wenn keine geeignete Position gefunden wurde, brechen wir die Schleife ab
                print(f"Error: Unable to place object {obj_name} after {max_attempts} attempts")
                continue

            for c in range(3):
                alpha = rotated_obj[:, :, 3] / 255.0
                image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] = \
                    rotated_obj[:, :, c] * alpha + image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] * (1.0 - alpha)

            # Calculate the four corners of the bounding box before rotation
            points = np.array([
                [0, 0],
                [obj_width, 0],
                [obj_width, obj_height],
                [0, obj_height]
            ], dtype=np.float32)

            # Apply the same transformation to the bounding box points
            transformed_points = cv2.transform(np.array([points]), M)[0]
            
            # Apply the offset
            transformed_points[:, 0] += x_offset
            transformed_points[:, 1] += y_offset

            x1, y1 = transformed_points[0]
            x2, y2 = transformed_points[1]
            x3, y3 = transformed_points[2]
            x4, y4 = transformed_points[3]

            labels.append((class_id, x1, y1, x2, y2, x3, y3, x4, y4))

    return image, labels

def calculate_class_percentages(object_files, excluded_classes, num_objects):
    available_classes = {int(os.path.basename(obj).split('_')[0]) for obj in object_files} - excluded_classes
    class_percentages = {class_id: 1 for class_id in available_classes}
    total_classes = len(class_percentages)
    percentage_per_class = 100 / total_classes
    class_percentages = {class_id: percentage_per_class for class_id in available_classes}
    return class_percentages

def read_statistics(file_path):
    class_percentages = {}
    class_positions = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        read_position = False
        for line in lines:
            if "Klasse" in line and "Mal gefunden" in line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = parts[1][:-1]
                    percentage = float(parts[-3].strip('%').replace(',', '.'))
                    class_percentages[class_id] = percentage
            elif "Durchschnittliche Position" in line:
                read_position = True
            elif read_position and "Klasse" in line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = parts[1][:-1]
                    mean_x = float(parts[3].strip('(,').replace(',', '.'))
                    mean_y = float(parts[4].strip(')').replace(',', '.'))
                    class_positions[class_id] = (mean_x, mean_y)
                    read_position = False

    print(class_percentages)
    return class_percentages, class_positions

def read_num_objects(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Mittelwert:" in line:
                parts = line.split()
                num_objects = float(parts[1])
                print(num_objects)
                return num_objects
    return 95  # Fallback-Wert, falls nichts gefunden wird

def read_num_objects_std(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Standardabweichung:" in line:
                parts = line.split()
                num_objects = float(parts[1])
                print(num_objects)
                return num_objects
    return 12  # Fallback-Wert, falls nichts gefunden wird

def generate_class_distribution_file(class_percentages, num_lines, output_file, mean, std_dev):
    total_percentage = sum(class_percentages.values())
    lines = []

    for _ in range(num_lines):
        line_distribution = {}
        num_objects = max(1, int(np.random.normal(mean, std_dev)))
        remaining_objects = num_objects
        
        # Berechnung der Grundverteilung basierend auf den Prozentsätzen
        for class_id, percentage in class_percentages.items():
            base_count = max(0, int(num_objects * (percentage / total_percentage)))
            line_distribution[class_id] = base_count
            remaining_objects -= base_count

        # Zufallsweise Verteilung der verbleibenden Objekte
        class_ids = list(class_percentages.keys())
        while remaining_objects > 0:
            class_id = random.choice(class_ids)
            line_distribution[class_id] += 1
            remaining_objects -= 1

        # Leichte Variation hinzufügen
        for class_id in line_distribution:
            if line_distribution[class_id] > 0:
                variation = random.randint(-2, 2)
                line_distribution[class_id] = max(0, line_distribution[class_id] + variation)

        lines.append(line_distribution)

    with open(output_file, 'w') as file:
        for line_distribution in lines:
            line = " ".join([f"Klasse {class_id}: {count}" for class_id, count in line_distribution.items()])
            file.write(line + "\n")

def verify_class_distribution(file_path, class_percentages, mean, std_dev, num_lines):
    total_lines = 0
    total_objects = 0
    cumulative_distribution = {class_id: 0 for class_id in class_percentages}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_distribution = {}
            parts = line.split()
            total_count = 0
            for i in range(0, len(parts), 3):
                class_id = int(parts[i+1][:-1])
                count = int(parts[i+2])
                line_distribution[class_id] = count
                total_count += count
                cumulative_distribution[class_id] += count

            total_lines += 1
            total_objects += total_count

    avg_objects = total_objects / total_lines if total_lines > 0 else 0
    print(f"Verification completed. Average objects per line: {avg_objects:.2f} (Expected: {mean} ± {std_dev})")

    # Verify the overall distribution
    for class_id, expected_percentage in class_percentages.items():
        actual_percentage = (cumulative_distribution[class_id] / total_objects) * 100
        print(f"Class {class_id}: {actual_percentage:.2f}% (Expected: {expected_percentage:.2f}%)")
