# CIFAR-100 Image Classification with ResNet-18

A complete PyTorch implementation for training a ResNet-18 model on the CIFAR-100 dataset with INT8 quantization support.

## Overview

This project implements a deep learning pipeline to classify images from the CIFAR-100 dataset into 100 fine-grained categories. It features:

- **ResNet-18 Architecture** - Proven CNN architecture optimized for CIFAR-100
- **Data Augmentation** - Random crops, horizontal flips for better generalization
- **Cosine Annealing LR Schedule** - Adaptive learning rate for optimal convergence
- **INT8 Quantization-Aware Training (QAT)** - Reduce model size by ~4x with minimal accuracy loss

## Project Structure

```
cifar100/
├── cifar100_train.py      # Main training script
├── cifar100_viewer.py     # Viewer for model predictions (displays 16 test images)
├── cifar-data/            # CIFAR-100 dataset directory
│   ├── train              # Training data (50,000 samples)
│   ├── test               # Test data (10,000 samples)
│   └── meta               # Metadata with class names
├── checkpoint.pth         # Training checkpoint (auto-saved each epoch)
├── cifar100_resnet18.pth  # Saved FP32 model (generated after training)
├── cifar100_resnet18_int8.pt  # Saved INT8 quantized model (generated after training)
├── PLAN.md                # Original project plan
├── PROMPT.md              # Original project prompt
└── README.md              # This file
```

## Requirements

- Python 3.7+
- PyTorch 2.0+ (CPU or CUDA version)
- torchvision
- NumPy

## Installation

1. Install PyTorch and dependencies:

```bash
# CPU version (current setup)
pip install torch torchvision numpy

# CUDA version (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

2. Ensure CIFAR-100 data is in `cifar-data/` directory:
   - `cifar-data/train` - Training pickle file
   - `cifar-data/test` - Test pickle file
   - `cifar-data/meta` - Metadata file

## Usage

### Training the Model

Download and extract CIFAR data to /cifar-data 
curl -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

Run the main training script:

```bash
python cifar100_train.py
```

This will:
1. Load and preprocess CIFAR-100 data
2. Train ResNet-18 for 100 epochs (FP32)
3. Apply Quantization-Aware Training (QAT) for 10 additional epochs
4. Convert the model to INT8 quantization
5. Save both FP32 and INT8 models

### Expected Output

```
============================================================
CIFAR-100 Training with ResNet-18
============================================================

Loading CIFAR-100 data...
Training samples: 50000
Test samples: 10000

Creating ResNet-18...
Model parameters: 11220132

Starting training...
Epoch [1/100] Loss: 3.9628 | Train: 8.94% | Test: 11.89%
...
Epoch [100/100] Loss: 0.1234 | Train: 95.67% | Test: 78.45%

FP32 model saved to cifar100_resnet18.pth

============================================================
INT8 Quantization-Aware Training (QAT)
============================================================

Starting QAT fine-tuning...
QAT Epoch 1/10 | Test Acc: 78.12%
...
QAT Epoch 10/10 | Test Acc: 77.89%

INT8 Test Accuracy: 77.89%
Accuracy drop: 0.56%

INT8 quantized model saved to cifar100_resnet18_int8.pt

============================================================
FINAL RESULTS
============================================================
FP32 Best Accuracy: 78.45%
INT8 Accuracy: 77.89%
Model size reduction: ~4x (FP32 -> INT8)
```

## Model Viewer

The project includes a viewer script to visualize model predictions on test images:

```bash
python cifar100_viewer.py
```

This will:
1. Detect available trained models (checkpoint.pth, cifar100_resnet18.pth, or cifar100_resnet18_int8.pt)
2. Prompt you to select which model to use
3. Display a 4x4 grid of 16 random test images with predictions
4. Show true labels, predicted labels, confidence scores, and overall accuracy
5. Color-code results (green = correct, red = incorrect)

### Loading Specific Model

You can also modify the script to load a specific model directly by changing the `selected_model` variable.

## Performance Expectations

| Model Type | Expected Accuracy | Model Size | Speed |
|------------|-------------------|------------|-------|
| FP32 (baseline) | 75-80% | 100% | 1x |
| INT8 (QAT) | 74-79% | ~25% | 2-4x faster |

## Using Trained Models

### Loading FP32 Model

```python
import torch
from cifar100_train import ResNet18

# Load model
model = ResNet18()
model.load_state_dict(torch.load('cifar100_resnet18.pth'))
model.eval()

# Make prediction
# ... (preprocess image and run inference)
```

### Loading INT8 Quantized Model

```python
import torch

# Load quantized model
model = torch.jit.load('cifar100_resnet18_int8.pt')
model.eval()

# Make prediction
# ... (preprocess image and run inference)
```

## Configuration

You can modify training parameters in `cifar100_train.py`:

```python
# Device configuration
device = torch.device('cpu')  # or torch.device('cuda') if GPU available

# Training hyperparameters
batch_size = 128
epochs = 100
lr = 0.1
```

## Dataset Information

CIFAR-100 contains:
- **100 classes** (fine labels)
- **20 superclasses** (coarse labels)
- **50,000 training images** (500 per class)
- **10,000 test images** (100 per class)
- **Image size**: 32x32 pixels, RGB

## Troubleshooting

### NumPy Deprecation Warning
You may see a warning about `dtype()` - this is harmless and doesn't affect training.

### Out of Memory
Reduce `batch_size` if you encounter memory issues:
```python
batch_size = 64  # or 32
```

### Slow Training
Training on CPU takes ~1-2 hours for 100 epochs. For faster training:
- Use a machine with NVIDIA GPU
- Install CUDA version of PyTorch
- The script automatically detects and uses GPU if available
- Mixed precision (AMP) is enabled automatically on CUDA devices

### CUDA Installation
```bash
# CUDA 12.1 (recommended for modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.6 (alternative)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Checkpoint Recovery

The training script automatically saves a checkpoint (`checkpoint.pth`) after every epoch. If training is interrupted, you can simply re-run the script and it will resume from the last saved epoch automatically.

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

- CIFAR-100 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- ResNet architecture by He et al. (2015)
- PyTorch team for the deep learning framework