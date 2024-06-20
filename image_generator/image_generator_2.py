import os
import glob
import cv2
from tqdm import tqdm
from helper_functions_2 import *

# Helferfunktionen definieren
def read_class_distribution(file_path):
    distributions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_counts = [int(count.split(': ')[1]) for count in line.split('Klasse ')[1:]]
            distributions.append(class_counts)
    return distributions


