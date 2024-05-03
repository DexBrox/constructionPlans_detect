# Iteration über jede Bilddatei
for image in image_files:
    results = model.predict([image], conf=0.1, show_labels=False, show_conf=False)
    for result in results:
        base_filename = os.path.basename(image)
        name, ext = os.path.splitext(base_filename)    

        new_filename = f"{name}_d{ext}"
        img_folder = os.path.join('/workspace/Yolov8/results', img_path)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        img_path_full = os.path.join(img_folder, new_filename)
        plot
        result.save(img_path_full)


        

        # Iteration durch jede Bounding Box und deren Klassen
        for box, cls in zip(result.obb.xyxyxyxy, result.obb.cls):
            output = (cls.item(),) + tuple(box.reshape(-1).tolist())
            output_list.append(output)

https://docs.ultralytics.com/modes/predict/#obb