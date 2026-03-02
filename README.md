# Lane Segmentation for Autonomous Driving

A semantic segmentation project designed to detect road lanes from images using PyTorch and `segmentation-models-pytorch`. 

This system uses a fully convolutional encoder-decoder architecture to generate high-precision lane masks, achieving an **80% Intersection over Union (IoU)** score. It is fully integrated with **DVC** for pipeline orchestration and **MLflow** for experiment tracking.

## 🌟 Features

- **Semantic Segmentation**: Pixel-level prediction of road lanes.
- **Custom Dataset Loading**: Implements a `LaneDataset` class with ImageNet normalization and horizontal flip augmentation.
- **Pre-trained Architectures**: Leverages robust encoder-decoder backbones (via `segmentation-models-pytorch`) initialized with ImageNet weights.
- **Experiment Tracking**: Hyperparameters, model versions, and IoU metrics are tracked automatically using MLflow.
- **Reproducible Pipeline**: Uses DVC (`dvc.yaml`) to cleanly separate training, evaluation, and logging stages.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, `segmentation-models-pytorch`
- **Computer Vision**: OpenCV, Albumentations
- **MLOps**: MLflow (Experiment Tracking), DVC (Data Version Control)
- **Language**: Python

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Lane-Segmentation.git
cd Lane-Segmentation

# Install dependencies (requires PyTorch and smp)
pip install torch torchvision segmentation-models-pytorch dvc mlflow PyYAML
```

## 🚀 Usage

The ML workflow is orchestrated by DVC. To execute the training pipeline:

```bash
# Start the local MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

# Run the DVC pipeline
dvc repro
```

You can view the MLflow UI at `http://localhost:5000` to monitor the training loss and validation IoU in real time.

## ⚙️ Configuration
All hyperparameters are managed centrally in `params.yaml`:
- Image resizing (`224x224`)
- Batch size and learning rate
- Optimizer settings
- MLflow experiment names