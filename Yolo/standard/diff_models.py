import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os
import yaml

# W&B Warnmeldungen unterdrücken
os.environ['WANDB_SILENT'] = 'true'

# Model and configuration setup
model_names = ['yolov10n.pt',
               'yolov9e.pt',
               'yolov8x-obb.pt'] 
data_name = 'Roewaplan_v2.yaml'
project_name = 'different_models'
config_yaml_name = 'config_min.yaml'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# Code-Start
main_folder = '/workspace/main_folder/'
configuration = {
    'data': os.path.join(main_folder, 'YAMLs', data_name),
    'project': os.path.join(main_folder, 'RESULTs', project_name),
}

for model_var in model_names:
    model_path = os.path.join(main_folder, 'MODELs', model_var)
    config_yaml_path = os.path.join(main_folder, 'CONFIGs', config_yaml_name)

    if not os.path.exists(config_yaml_path):
        os.makedirs(config_yaml_path)

    # Load configuration file
    with open(config_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        base_config = config['base_config']

    # Initialize wandb
    wandb.init(project=f"Masterarbeit_{project_name}_{os.path.splitext(data_name)[0]}", name=os.path.splitext(model_var)[0])

    # Load YOLO model
    model = YOLO(model_path).to(device)

    # Train the model
    model.train(**base_config, **configuration)

    # Export the model to ONNX format
    model.export(format='onnx')

    # Finish wandb session
    wandb.finish()
