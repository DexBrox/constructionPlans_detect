import os
from PIL import Image
import numpy as np

def get_image_sizes(directory):
    image_sizes = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            print(f"Checking file: {f}")  # Add debug information
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff')):
                fp = os.path.join(dirpath, f)
                try:
                    with Image.open(fp) as img:
                        width, height = img.size
                        image_sizes.append((f, width, height))
                        print(f"{f}: {width}x{height}")  # Add debug information
                except (FileNotFoundError, PermissionError, OSError) as e:
                    print(f"File '{fp}' not found, no permission, or not readable, skipping. Error: {e}")
    return image_sizes

def save_image_sizes(image_sizes, output_file):
    widths = [width for _, width, _ in image_sizes]
    heights = [height for _, _, height in image_sizes]

    mean_width = np.mean(widths) if widths else 0
    stddev_width = np.std(widths) if widths else 0
    mean_height = np.mean(heights) if heights else 0
    stddev_height = np.std(heights) if heights else 0

    with open(output_file, 'w') as f:
        for image_name, width, height in image_sizes:
            f.write(f"{width}x{height}\n")
        
        f.write(f"\nMeans:\nWidth: {mean_width:.2f}\nHeight: {mean_height:.2f}\n")
        f.write(f"\nStandard Deviations:\nWidth: {stddev_width:.2f}\nHeight: {stddev_height:.2f}\n")

if __name__ == "__main__":
    directory = "/workspace/image_generator/background_rp_v3/"  # Fixed directory
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
    else:
        image_sizes = get_image_sizes(directory)
        output_file = "image_sizes.txt"  # Set output file
        save_image_sizes(image_sizes, output_file)
        print(f"Image sizes have been saved in '{output_file}'.")
        if not image_sizes:
            print("No images were found.")
