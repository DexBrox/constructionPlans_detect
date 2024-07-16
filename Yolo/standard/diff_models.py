import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import os
import yaml

# W&B Warnmeldungen unterdrücken
os.environ['WANDB_SILENT'] = 'true'

# Model and configuration setup
model_names = [
    #'yolov10n.pt', 
    #'yolov10s.pt', 
    #'yolov10m.pt', 
    #'yolov10b.pt', 
    #'yolov10l.pt', 
    #'yolov10x.pt',

    'yolov9t.pt', 
    'yolov9s.pt', 
    'yolov9m.pt', 
    'yolov9c.pt', 
    'yolov9e.pt',

    #'yolov8n-obb.pt', 
    #'yolov8s-obb.pt', 
    #'yolov8m-obb.pt', 
    #'yolov8l-obb.pt', 
    #'yolov8x-obb.pt'
]
data_name = 'Roewaplan_v3.yaml'
project_name = 'different_models'
config_yaml_name = 'final_config_best_v9e'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Code-Start
main_folder = '/workspace/main_folder/'
configuration = {
    'data': os.path.join(main_folder, 'YAMLs', data_name),
    'project': os.path.join(main_folder, 'RESULTs', project_name),
}

for model_var in model_names:
    try:
        model_path = os.path.join(main_folder, 'MODELs', model_var)
        config_yaml_path = os.path.join(main_folder, 'CONFIGs', config_yaml_name)

        if not os.path.exists(config_yaml_path):
            os.makedirs(config_yaml_path)

        # Load configuration file
        with open(config_yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            base_config = config['base_config']

        # Initialize wandb
        wandb.init(project=f"FINAL_Masterarbeit_{project_name}_{os.path.splitext(data_name)[0]}", name=os.path.splitext(model_var)[0])

        # Load YOLO model
        model = YOLO(model_path).to(device)

        # Train the model
        model.train(**base_config, **configuration) #

        # Export the model to ONNX format
        model.export(format='onnx')

        # Finish wandb session
        wandb.finish()
    except Exception as e:
        print(f"Fehler bei Modell {model_var}: {e}")
        wandb.finish()
        continue
