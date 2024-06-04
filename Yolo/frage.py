# Basis-Konfiguration für YOLO
base_config = {
    'model': '/workspace/main_folder/models/yolov8x-obb.pt',
    'data': 'YAML/Roewaplan.yaml',
    'project': '/workspace/Yolov8/standard/results/results_variation_theo',
    'epochs': 400,
    'patience': 1000,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'momentum': 0.937,
}

# Definition der verschiedenen Konfigurationsvariationen
config_variation_dropout = [
    {'dropout': 0.35},
    {'dropout': 0.3},
    {'dropout': 0.25}
]

config_variation_batch = [
    {'batch': 2},
    {'batch': 3},
    {'batch': 4}
]

config_variation_lr0 = [
    {'lr0': 0.008},
    {'lr0': 0.01},
    {'lr0': 0.012}
]

config_variation_imgsz = [
    {'imgsz': 1280},
    {'imgsz': 1400},
    {'imgsz': 1500}
]

# Definition der verschiedenen Konfigurationsvariationen
config_variations = {
    'var_dropout': config_variation_dropout,
    'var_batch': config_variation_batch,
    'var_lr0': config_variation_lr0,
    'var_imgsz': config_variation_imgsz,
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for variation_name, configs in config_variations.items():
    for config in configs:
        variable_parameter = f"{config[variation_name.split('_')[1]]}"
        run_name = f"theo_{variable_parameter}"
        color_index = configs.index(config) % len(colors)
        wandb.init(project="Masterarbeit_dataset-original_variation_dblim2", name=run_name, config={"run_color": colors[color_index]})

        # Erstelle eine kombinierte Konfiguration für das Training
        train_config = {**base_config, **config}

        # Initialisiere das YOLO-Modell
        model = YOLO(train_config['model']).to(device)

        # Trainiere das Modell
        try:
            model.train(
                **train_config,  # Spread operator, um alle Schlüssel-Wert-Paare zu übergeben
                device=device
            )

            # Validiere und exportiere das Modell falls nötig
            model.val()
            model.export(format='onnx')

        except Exception as e:
            print(f"Fehler beim Training mit Konfiguration: {config}")
            print(f"Fehlermeldung: {e}")

        finally:
            # Beende den aktuellen W&B Run
            wandb.finish()
