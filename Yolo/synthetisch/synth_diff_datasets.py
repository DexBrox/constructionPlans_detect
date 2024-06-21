import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
import torch
import yaml
import tempfile
import os


# Model and configuration setup
model_name = 'yolov8x-obb.pt'
project_name = 'synth_diff_datasets_v3'
config_yaml_name = 'config_best_short.yaml'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

main_folder = '/workspace/main_folder/'
model_path = main_folder + 'MODELs/' + model_name
config_yaml_path = main_folder + 'CONFIGs/' + config_yaml_name

# Load base configuration file
with open(config_yaml_path, 'r') as file:
    config = yaml.safe_load(file)
    base_config = config['base_config']

# List of dataset base paths
dataset_base_paths = [f'/workspace/datasets/synth/synth_v3_{i}' for i in range(1, 17)]

for base_path in dataset_base_paths:
    train_path = os.path.join(base_path, 'images/train')
    val_path = os.path.join(base_path, 'images/val')

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"Verzeichnis fehlt in {base_path}. Überspringe diesen Datensatz.")
        continue

    data_yaml_content = {
        'path': base_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'Rohbau',
            1: 'Kunststoff',
            2: 'Dämmung',
            3: 'Dübel',
            4: 'Schraube',
            5: 'Alublech',
            6: 'Stahlblech',
            7: 'Beschriftung',
            8: 'Folie',
            9: 'PfostenRiegel',
            10: 'Konsole',
            11: 'Edelstahl',
            12: 'Glas',
            13: 'Dichtung',
            14: 'Holz',
            15: 'Systemprofil'
        }
    }

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as temp_yaml:
        yaml.dump(data_yaml_content, temp_yaml)
        temp_yaml_path = temp_yaml.name

    configuration = {
        'data': temp_yaml_path,
        'project': main_folder + 'RESULTs/' + project_name,
    }

    # Initialize wandb for each dataset
    run_name = os.path.basename(base_path)
    wandb.init(project=f"Masterarbeit_{project_name}_{os.path.splitext(model_name)[0]}", name=run_name)

    # Log the dataset being used
    wandb.log({"dataset": data_yaml_content['path']})

    # Load YOLO model
    model = YOLO(model_path).to(device)

    # Train the model
    model.train(**base_config, **configuration, model=model_path)

    # Clean up temporary file
    os.remove(temp_yaml_path)

    # Finish the wandb run for this dataset
    wandb.finish()

# Export the model to ONNX format
model.export(format='onnx')
