"""
CIFAR-100 Model Viewer
Loads trained model and displays 16 random test images with predictions
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os


# ===================== RESNET ARCHITECTURE (same as training script) =====================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# ===================== DATA LOADING =====================

class CIFAR100PickleDataset(Dataset):
    """Custom Dataset for loading CIFAR-100 from pickle files"""
    def __init__(self, pickle_file, transform=None):
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        self.data = data_dict[b'data']
        self.labels = data_dict[b'fine_labels']
        self.coarse_labels = data_dict.get(b'coarse_labels', [0] * len(self.labels))
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        arr = self.data[idx].reshape(3, 32, 32).astype('float32')
        img = torch.from_numpy(arr) / 255.0
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# CIFAR-100 class names (fine labels)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plate', 'plum', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 
    'worm'
]


def load_model(model_path='cifar100_resnet18.pth', device='cuda'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
    
    # Check if it's a quantized model (JIT)
    if model_path.endswith('.pt'):
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Loaded quantized model from {model_path}")
        return model, device
    
    # Load checkpoint or state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ResNet18()
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # It's a checkpoint file
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"Loaded model from checkpoint (epoch {epoch})")
    else:
        # Assume it's a direct state dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Loaded FP32 model from {model_path}")
    return model, device


def get_test_images(num_images=16, test_path='cifar-data/test'):
    """Load test dataset and randomly select images"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    dataset = CIFAR100PickleDataset(test_path, transform=transform)
    
    # Randomly select indices
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    
    return images, labels


def denormalize(tensor, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)):
    """Denormalize image tensor for display"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def predict_and_display(model, device, num_images=16):
    """Predict and display images in 4x4 grid"""
    # Load test images
    images, true_labels = get_test_images(num_images)
    
    # Prepare batch for model
    batch = torch.stack(images).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(batch)
        _, predicted = outputs.max(1)
        probabilities = F.softmax(outputs, dim=1)
        max_probs = probabilities.max(1)[0]
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(true_labels, predicted.cpu()) if t == p.item())
    accuracy = 100 * correct / num_images
    
    # Create 4x4 grid (50% smaller than original 14x14 → 7x7)
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    fig.suptitle('CIFAR-100 Model Predictions', fontsize=11, fontweight='bold', y=0.99)
    
    for idx in range(num_images):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # Denormalize and convert to numpy for display
        img = denormalize(images[idx])
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Display image
        ax.imshow(img_np)
        
        # Get prediction info
        true_label = true_labels[idx]
        pred_label = predicted[idx].item()
        confidence = max_probs[idx].item()
        
        true_name = CIFAR100_CLASSES[true_label]
        pred_name = CIFAR100_CLASSES[pred_label]
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        # Set title with prediction overlay (readable font for smaller window)
        title = f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}"
        ax.set_title(title, color=color, fontsize=8, pad=2)
        ax.axis('off')
    
    # Add summary text box at the bottom (better positioning and readability)
    summary_text = f"Overall Accuracy: {correct}/{num_images} ({accuracy:.1f}%)"
    fig.text(0.5, 0.03, summary_text, ha='center', va='bottom', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.4),
             fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.show()
    
    # Print summary to console as well
    print(f"\nAccuracy on {num_images} random images: {correct}/{num_images} ({accuracy:.1f}%)")


def main():
    print("=" * 60)
    print("CIFAR-100 Model Viewer")
    print("=" * 60)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find available models
    available_models = []
    
    if os.path.exists('checkpoint.pth'):
        available_models.append(('checkpoint.pth', 'Training Checkpoint (resume capable)'))
    
    if os.path.exists('cifar100_resnet18.pth'):
        available_models.append(('cifar100_resnet18.pth', 'Final FP32 Model'))
    
    if os.path.exists('cifar100_resnet18_int8.pt'):
        available_models.append(('cifar100_resnet18_int8.pt', 'Quantized INT8 Model'))
    
    if not available_models:
        print("\nError: No model files found.")
        print("Please train the model first.")
        print("Expected: checkpoint.pth, cifar100_resnet18.pth, or cifar100_resnet18_int8.pt")
        return
    
    # Display available models
    print("\nAvailable models:")
    for idx, (filename, description) in enumerate(available_models):
        print(f"  [{idx + 1}] {filename} - {description}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nSelect model to load (enter number): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx][0]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Load selected model
    model, device = load_model(selected_model, device)
    
    if model is None:
        return
    
    # Check test data
    if not os.path.exists('cifar-data/test'):
        print("Error: Test data not found at cifar-data/test")
        return
    
    print(f"\nDisplaying 16 random test images with predictions from {selected_model}...")
    predict_and_display(model, device, num_images=16)


if __name__ == '__main__':
    main()