import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os
import yaml
import setproctitle

# Model and configuration setup
model_name = 'yolov8x-obb.pt'
data_name = 'Roewaplan_v3.yaml'
project_name = 'allforcomparison'
config_yaml_name = 'config_best_v8x-obb.yaml'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# Setze den benutzerdefinierten Prozessnamen
setproctitle.setproctitle('VJ_' + project_name + '_' + os.path.basename(model_name) + '_VJ_write_me_if_gpu_needed')

main_folder = '/workspace/main_folder/'
configuration = {
    'data': os.path.join(main_folder, 'YAMLs', data_name),
    'project': os.path.join(main_folder, 'RESULTs', project_name),
}
model_path = os.path.join(main_folder, 'MODELs', model_name)
config_yaml_path = os.path.join(main_folder, 'CONFIGs', config_yaml_name)

if not os.path.exists(os.path.dirname(config_yaml_path)):
    os.makedirs(os.path.dirname(config_yaml_path))

# Load configuration file
with open(config_yaml_path, 'r') as file:
    config = yaml.safe_load(file)
    base_config = config['base_config']

# Initialize wandb
wandb.init(project=f"FINAL_BIG_Masterarbeit_{project_name}_{os.path.splitext(model_name)[0]}", name=os.path.splitext(data_name)[0])

# Load YOLO model
model = YOLO(model_path).to(device)

# Train the model
model.train(**base_config, **configuration, model=model_path)

# Export the model to ONNX format
model.export(format='onnx')

# Finish wandb session
wandb.finish()
