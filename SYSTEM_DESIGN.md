# System Design: Lane Segmentation

## 1. Overview
The **Lane Segmentation** system is designed for autonomous driving applications to accurately detect and map road lanes from front-facing vehicle camera images. It treats lane detection as a pixel-wise semantic segmentation problem.

## 2. Core Architecture

The architecture utilizes the `segmentation-models-pytorch` (SMP) library, allowing the system to flexibly attach state-of-the-art CNN/Transformer encoders to various segmentation decoders (like UNet, FPN, DeepLabV3).

### Pipeline Stages
1. **Data Ingestion (`LaneDataset`)**:
   - Reads raw RGB images and their corresponding binary masks (where lanes = 1, background = 0).
   - Applies resizing to $224 	imes 224$ for uniform batch processing.
   - Applies Albumentations-based data augmentation (e.g., horizontal flipping, random brightness/contrast) to improve generalization across different lighting conditions and road curves.
   - Normalizes images using ImageNet statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

2. **Model Architecture**:
   - **Encoder**: Pre-trained ResNet/EfficientNet backbone initialized with ImageNet weights. Acts as the feature extractor.
   - **Decoder**: UNet or Feature Pyramid Network (FPN) architecture. Upsamples the encoded feature maps back to the original image dimensions.
   - **Output Head**: A $1 	imes 1$ convolution predicting the probability of a pixel belonging to the "lane" class.

3. **Training & Orchestration**:
   - The training loop is written in standard PyTorch, utilizing Binary Cross Entropy (BCE) with Logits Loss or Dice Loss.
   - **DVC (Data Version Control)** manages the execution graph. The `dvc.yaml` file defines the `training` stage, ensuring it only re-runs when `params.yaml` or source code changes.
   - **MLflow Tracking**: Integrated directly into the training loop to capture metrics (IoU, Epoch Loss) and store model artifacts.

## 3. Design Choices & Trade-offs
* **Pre-trained Encoders**: Training a segmentation network from scratch on small datasets usually leads to overfitting. Leveraging pre-trained ImageNet weights provides robust low-level edge and texture detectors right out of the box, drastically reducing convergence time.
* **ImageNet Normalization**: Since the encoder weights are pre-trained on ImageNet, we must align our input distribution to match what the encoder saw during its initial training.
* **Intersection over Union (IoU) Metric**: Pixel accuracy is highly misleading for lane detection because lanes occupy <5% of the total image pixels (a model guessing "no lane" everywhere would achieve >95% accuracy). IoU directly measures the overlap between predicted lane pixels and actual lane pixels.
* **DVC + MLflow Setup**: Instead of writing standalone scripts, wrapping the project in DVC makes the pipeline DAG-based and reproducible, making hyperparameter sweeps in `params.yaml` trivial.

## 4. Future Enhancements
* **Real-Time Inference via TensorRT**: Export the PyTorch model to ONNX, then compile with TensorRT to achieve 60+ FPS on edge devices (like NVIDIA Jetson).
* **Instance Segmentation**: Moving from semantic (all lanes are the same class) to instance segmentation (differentiating between the ego-lane, adjacent lanes, and opposite lanes) using Mask R-CNN.
* **Temporal Consistency**: Adding an LSTM/GRU layer at the bottleneck to smooth predictions across sequential video frames, reducing flickering.
