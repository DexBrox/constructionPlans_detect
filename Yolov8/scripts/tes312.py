from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg'
results = model(['bus.jpg', 'zidane.jpg'])  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f'results{i}.jpg')










# Iteration über jede Bilddatei
for image in image_files:
    results = model.predict([image], conf=0.1, show_labels=False, show_conf=False)

    # Speichern der Ergebnisbilder
    for result in results:
        base_filename = os.path.basename(image)
        name, ext = os.path.splitext(base_filename)    
        new_filename = f"{name}_d{ext}"
        img_folder = os.path.join('/workspace/Yolov8/results', img_path)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        img_path_full = os.path.join(img_folder, new_filename)

        result.save(img_path_full)

        # Iteration durch jede Bounding Box und deren Klassen
        for box, cls in zip(result.obb.xyxyxyxy, result.obb.cls):
            output = (cls.item(),) + tuple(box.reshape(-1).tolist())
            output_list.append(output)