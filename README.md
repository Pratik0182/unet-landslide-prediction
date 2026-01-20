# Landslide Prediction
PyTorch implementation of deep learning based landslide prediction using U-Net architecture in convolution neural network.

<p align="center">
  <img src="utils/results.gif" alt="monodepth">
</p>

**Landslide Prediction model with multi-spectral satellite(14 channels) imagery.**

---

## Requirements
This code was tested with Python 3.12.7, PyTorch with CUDA drivers downloaded and Microsoft Windows 25H2.  
Training takes about **1-2 hours** with the default parameters (50 epochs, batch size 32, learning rate 1e-4) on a standard GPU-enabled machine.

## I just want to try it on an image!
There is a test mode in `landslide_main.py` which allows you to quickly run the model on test data. Make sure you have a trained model in the `models/` directory.
```powershell
python src/landslide_main.py --mode test --data_path "path/to/dataset" --model_path "models/unet_landslide_torch.pth"
```

## Data
The model expects HDF5 (`.h5`) files containing multi-spectral images (14 channels) and binary masks.
The dataset should be organized as follows:
```text
Dataset/
└── archive/
    ├── TrainData/
    │   ├── img/
    │   └── mask/
    └── TestData/
        └── img/
```
You can find the specific data splits in the [filenames](utils/filenames) folder. To generate or refresh these splits, run:
```powershell
python utils/generate_filenames.py
```

## Training
The model's dataloader expects a data folder path as well as the generated filename lists.
```powershell
python src/landslide_main.py --mode train --data_path "path/to/dataset" --epochs 50 --batch_size 32 --learning_rate 1e-4
```
You can continue training by loading a checkpoint:
```powershell
python src/landslide_main.py --mode train --data_path "path/to/dataset" --checkpoint_path "models/unet_landslide_torch.pth"
```

## Testing
To generate predictions on the test set, use the `test` mode. The network will output `.h5` masks in the specified output directory.
```powershell
python src/landslide_main.py --mode test --data_path "path/to/dataset" --model_path "models/unet_landslide_torch.pth" --output_path "predictions"
```

## Evaluation
To calculate detailed metrics (Precision, Recall, F1-Score) on the validation split, run:
```powershell
python utils/evaluate.py --data_path "path/to/dataset" --model_path "models/unet_landslide_torch.pth"
```

## Models
Our pre-trained model is available directly in the `models/` directory.
All our models were trained for 50 epochs, 128x128 resolution and a batch size of 32, please see our project details for more information.
We used HDF5 format for both images and masks during training.

Here are the models available:
* `unet_landslide_torch.pth`: Our main model trained on the landslide dataset.

## Project Structure
- `src/`: Core source code
  - `landslide_dataloader.py`: Custom dataset for loading HDF5 images and masks.
  - `landslide_model.py`: U-Net model architecture.
  - `landslide_trainer.py`: Training and validation epoch logic.
  - `landslide_main.py`: Main script for training and testing.
- `utils/`: Toolbox and helper scripts
  - `filenames/`: Text files listing training and validation file names.
  - `metrics.py`: Loss functions and performance metrics.
  - `evaluate.py`: Script for detailed model evaluation.
- `models/`: Directory to store trained model weights.
- `predictions/`: Directory to store prediction results.

## License
This project is licensed under the terms provided in the [LICENSE](LICENSE) file.
