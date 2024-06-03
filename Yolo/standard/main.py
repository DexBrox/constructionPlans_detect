import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os
import yaml

# Model and configuration setup
model_name = 'yolov8x-obb.pt'
data_name = 'Theo.yaml'
project_name = 'standard'
config_yaml_name = 'config_best.yaml'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

main_folder = '/workspace/main_folder'
configuration = {
    'data': main_folder + data_name,
    'project': main_folder + project_name,
}
model = main_folder + model_name
config_yaml = main_folder + config_yaml_name

# Load configuration file
with open(config_yaml, 'r') as file:
    config = yaml.safe_load(file)
    base_config = config['base_config']

# Initialize wandb
wandb.init(project=f"Masterarbeit_{project_name}_{os.path.splitext(model_name)[0]}", name=os.path.splitext(data_name)[0])

# Load YOLO model
model = YOLO(model).to(device)

# Train the model
model.train(**base_config, **configuration, model=model)

# Export the model to ONNX format
model.export(format='onnx')

# Finish wandb session
wandb.finish()
