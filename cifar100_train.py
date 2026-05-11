"""
CIFAR-100 Training with ResNet-18 and INT8 Quantization-Aware Training
Complete, fixed implementation
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import time
import os


# ===================== DATA LOADING =====================

class CIFAR100PickleDataset(Dataset):
    """Custom Dataset for loading CIFAR-100 from pickle files"""
    def __init__(self, pickle_file, transform=None):
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        self.data = data_dict[b'data']
        self.labels = data_dict[b'fine_labels']
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Reshape and normalize to [0, 1]
        arr = self.data[idx].reshape(3, 32, 32).astype('float32')
        img = torch.from_numpy(arr) / 255.0
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_data_loaders(train_path, test_path, batch_size=128, num_workers=2):
    """Create training and testing data loaders"""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = CIFAR100PickleDataset(train_path, transform=train_transform)
    test_dataset = CIFAR100PickleDataset(test_path, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


# ===================== RESNET ARCHITECTURE =====================

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


# ===================== CHECKPOINT FUNCTIONS =====================

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, scaler=None, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, filename)
    print("Checkpoint saved: {}".format(filename))


def load_checkpoint(model, optimizer, scheduler, scaler=None, filename='checkpoint.pth'):
    """Load training checkpoint. Returns (start_epoch, best_acc) or (0, 0) if no checkpoint."""
    if os.path.exists(filename):
        print("\nLoading checkpoint: {}".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        best_acc = checkpoint['best_acc']
        print("Resumed from epoch {}. Best accuracy so far: {:.2f}%".format(start_epoch, best_acc))
        return start_epoch, best_acc
    else:
        print("\nNo checkpoint found. Starting training from scratch.")
        return 0, 0


# ===================== TRAINING FUNCTIONS =====================

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None and device.type == 'cuda':
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


# ===================== MAIN =====================

def main():
    # Device configuration - auto-detect GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    if device.type == 'cuda':
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print("WARNING: No GPU detected. Training will be slow on CPU.")
        print("For faster training, install CUDA-enabled PyTorch.")
    
    batch_size = 128
    epochs = 100
    lr = 0.1
    
    print("=" * 60)
    print("CIFAR-100 Training with ResNet-18")
    print("=" * 60)
    
    # Check data files
    train_path = 'cifar-data/train'
    test_path = 'cifar-data/test'
    
    if not os.path.exists(train_path):
        print("Error: Training data not found at {}".format(train_path))
        return
    if not os.path.exists(test_path):
        print("Error: Test data not found at {}".format(test_path))
        return
    
    # Load data
    print("\nLoading CIFAR-100 data...")
    train_loader, test_loader = get_data_loaders(train_path, test_path, batch_size, num_workers=4)
    print("Training samples: {}".format(len(train_loader.dataset)))
    print("Test samples: {}".format(len(test_loader.dataset)))
    
    # Create model
    print("\nCreating ResNet-18...")
    model = ResNet18()
    model.to(device)
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler (only used with GPU)
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("Mixed precision (AMP) enabled")
    
    # Load checkpoint if exists (resume training)
    start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, scaler)
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    last_print_time = start_time
    print_interval = 10  # seconds
    
    for epoch in range(start_epoch, epochs):
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()
        
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        
        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, best_acc, scaler)
        
        # Print progress every 10 seconds or every 10 epochs (whichever comes first)
        current_time = time.time()
        if (current_time - last_print_time >= print_interval) or (epoch + 1) % 10 == 0 or epoch == start_epoch:
            elapsed_so_far = current_time - start_time
            print("Epoch [{}/{}] Loss: {:.4f} | Train: {:.2f}% | Test: {:.2f}% | Time: {:.1f}s".format(
                epoch+1, epochs, loss, train_acc, test_acc, elapsed_so_far))
            last_print_time = current_time
    
    elapsed = time.time() - start_time
    print("\nTraining completed in {:.2f} seconds".format(elapsed))
    print("Best Test Accuracy: {:.2f}%".format(best_acc))
    
    # Save FP32 model
    torch.save(model.state_dict(), 'cifar100_resnet18.pth')
    print("\nFP32 model saved to cifar100_resnet18.pth")
    
    # ===================== INT8 QUANTIZATION (QAT) =====================
    print("\n" + "=" * 60)
    print("INT8 Quantization-Aware Training (QAT)")
    print("=" * 60)
    
    try:
        # Prepare model for QAT
        model.train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig()
        torch.ao.quantization.prepare_qat(model, inplace=True)
        
        # QAT fine-tuning (fewer epochs)
        qat_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        qat_criterion = nn.CrossEntropyLoss()
        
        print("\nStarting QAT fine-tuning...")
        for epoch in range(10):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                qat_optimizer.zero_grad()
                outputs = model(inputs)
                loss = qat_criterion(outputs, targets)
                loss.backward()
                qat_optimizer.step()
            
            qat_acc = evaluate(model, test_loader, device)
            print("QAT Epoch {}/10 | Test Acc: {:.2f}%".format(epoch+1, qat_acc))
        
        # Convert to INT8
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)
        int8_acc = evaluate(model, test_loader, device)
        print("\nINT8 Test Accuracy: {:.2f}%".format(int8_acc))
        print("Accuracy drop: {:.2f}%".format(best_acc - int8_acc))
        
        # Save quantized model
        scripted_model = torch.jit.script(model)
        scripted_model.save('cifar100_resnet18_int8.pt')
        print("\nINT8 quantized model saved to cifar100_resnet18_int8.pt")
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print("FP32 Best Accuracy: {:.2f}%".format(best_acc))
        print("INT8 Accuracy: {:.2f}%".format(int8_acc))
        print("Model size reduction: ~4x (FP32 -> INT8)")
        
    except Exception as e:
        print("\nQuantization not available: {}".format(e))
        print("FP32 model saved successfully.")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()