import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os
import yaml
import setproctitle

# Model and configuration setup
model_name = 'yolov9e.pt'
data_name = 'Roewaplan_v3.yaml'
project_name = 'hyperparameter_test'
config_yaml_name = 'config_best.yaml'
device = 'cuda:2'


# Setze den benutzerdefinierten Prozessnamen
setproctitle.setproctitle('VJ_' + project_name + '_' + os.path.basename(model_name))

main_folder = '/workspace/main_folder/'
configuration = {
    'data': main_folder + 'YAMLs/' + data_name,
    'project': main_folder + 'RESULTs/' + project_name,
}
model_path = main_folder + 'MODELs/' + model_name
config_yaml = main_folder + 'CONFIGs/' + config_yaml_name

if not os.path.exists(config_yaml):
    os.makedirs(config_yaml)

# Load configuration file
with open(config_yaml, 'r') as file:
    config = yaml.safe_load(file)
    base_config = config['base_config']

# Define hyperparameter variations from smallest to largest
variations = {
    # 'imgsz': [320, 640, 960, 1280, 1600, 1920, 2040],
    # 'batch': [1, 2, 4, 8, 16],
    # 'batch': [4],
    # 'lr0': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'optimizer': ['Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'], #'SGD', 
    'warmup_epochs': [0, 3, 5, 10, 20],

    'dropout': [0.0, 0.2, 0.4, 0.6, 0.8],
    
    #'hsv_h': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #'hsv_s': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #'hsv_v': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #'translate': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

    #'scale': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #'fliplr': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #'auto_augment': ['randaugment', 'autoaugment', 'augmix'],
    #'crop_fraction': [0.1, 0.3, 0.55, 0.8, 1.0],
    #'mosaic': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

}

# Function to train with a single hyperparameter variation
def train_with_variation(parameter, values):
    for value in values:
        try:
            run_name = f"{parameter}_{value}"
            wandb.init(project=f"FINAL_Masterarbeit_{project_name}_{os.path.splitext(model_name)[0]}", name=run_name, reinit=True)

            # Update the configuration with the current hyperparameter
            current_config = base_config.copy()
            current_config[parameter] = value

            # Load YOLO model
            model = YOLO(model_path).to(device)

            # Train the model with current configuration
            model.train(**current_config, **configuration, model=model_path)

            # Export the model to ONNX format
            model.export(format='onnx')

            # Finish wandb session
            wandb.finish()

        except Exception as e:
            print(f"Skipping {run_name} due to error: {e}")
            wandb.finish()

# Iterate over each hyperparameter and its values
for param, values in variations.items():
    train_with_variation(param, values)
