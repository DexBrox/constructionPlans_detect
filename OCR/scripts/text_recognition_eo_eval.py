import itertools
import cv2
import easyocr
import os

# Importieren Sie Ihre evaluate_cer Funktion
from cer import evaluate_cer

# Parameterbereiche
scale_factors = [1.0, 2.0]
decoders = ['greedy', 'beamsearch']
text_thresholds = [0.3, 0.4, 0.5]
low_texts = [0.4, 0.5]
link_thresholds = [0.1, 0.2, 0.3]
canvas_sizes = [1024, 1280]
mag_ratios = [1.0, 1.5, 2.0]

# Konstante Parameter
languages = ['de', 'en']
gpu = True
workers = 10
batch_size = 10
paragraph = False
rotation_info = [0, 90, 180, 270]

def process_image_easyocr(image_path, scale_factor, decoder, text_threshold, low_text, link_threshold, canvas_size, mag_ratio):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Failed to load image at path: {image_path}")

    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    reader = easyocr.Reader(languages, gpu=gpu)
    result = reader.readtext(scaled_image, detail=1, paragraph=paragraph, workers=workers, batch_size=batch_size, rotation_info=rotation_info,
                             low_text=low_text, link_threshold=link_threshold, canvas_size=canvas_size, mag_ratio=mag_ratio, text_threshold=text_threshold)
    return result

# Generieren aller Parameterkombinationen
param_combinations = list(itertools.product(scale_factors, decoders, text_thresholds, low_texts, link_thresholds, canvas_sizes, mag_ratios))

best_cer = float('inf')
best_params = None

image_path = '../pdf/img/1.jpg'  # Pfad zur Testbild-Datei
gt_path = '../../labels'  # Pfad zum Ground Truth Verzeichnis
output_dir = '../results/txt'  # Ausgabeverzeichnis für Texterkennungsergebnisse

for params in param_combinations:
    # Ausführen der Texterkennung mit den aktuellen Parametern
    results = process_image_easyocr(image_path, *params)
    
    # Hier sollten Sie die Ergebnisse in einer Form speichern, die von Ihrer evaluate_cer Funktion verarbeitet werden kann
    
    # Berechnen der CER für die aktuellen Ergebnisse
    cer = evaluate_cer(gt_path, output_dir, 1)  # Der dritte Parameter hängt von Ihrer Implementierung von evaluate_cer ab
    
    if cer < best_cer:
        best_cer = cer
        best_params = params

print(f"Best CER: {best_cer} with parameters {best_params}")
