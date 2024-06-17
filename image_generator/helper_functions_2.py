# Helferfunktionen für die Bildgenerierung
def do_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def place_objects_in_image_ft(background_files, objects, image_height, image_width, distribution, rotation_range, scale_range, allow_overlap, use_backgrounds):
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

    class_counts = {i: distribution[i] for i in range(len(distribution))}

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
