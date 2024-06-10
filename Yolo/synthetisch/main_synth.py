import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os
import yaml

# Model and configuration setup
model_name = 'yolov8x-obb.pt'
data_name = 'TRAIN_synth_VAL_rp_64k.yaml'
project_name = 'synth_ft'
config_yaml_name = 'config_best.yaml'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

main_folder = '/workspace/main_folder/'
configuration = {
    'data': main_folder + 'YAMLs/' + data_name,
    'project': main_folder + 'RESULTs/' + project_name,
}
model = main_folder + 'MODELs/' + model_name
config_yaml = main_folder + 'CONFIGs/' + config_yaml_name

if not os.path.exists(config_yaml):
    os.makedirs(config_yaml)

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
