from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_dir, output_dir):
    image_paths = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                if filename.endswith(".pdf"):
                    filename = filename[:-4] + ""
                image_path = f'{output_dir}/{filename}.jpg'
                image.save(image_path, 'JPEG')
                image_paths.append(image_path)
    return image_paths
