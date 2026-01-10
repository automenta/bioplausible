"""
Utilities for mechanics demos - MNIST loading, training helpers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def load_mnist_subset(train=True, subset_size=1000, batch_size=64):
    """
    Load a small subset of MNIST for fast demos.
    
    Args:
        train: If True, load training set, else test set
        subset_size: Number of samples to use (for speed)
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader with MNIST subset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    dataset = datasets.MNIST(
        './data', 
        train=train, 
        download=True,
        transform=transform
    )
    
    # Create subset for speed
    if subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_single_digit(digit=5):
    """Get a single MNIST image for visualization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Find first occurrence of digit
    for img, label in dataset:
        if label == digit:
            return img.unsqueeze(0), label
    
    return dataset[0][0].unsqueeze(0), dataset[0][1]


def train_quick_epoch(model, dataloader, optimizer, device='cpu'):
    """
    Train for one quick epoch.
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluate model accuracy.
    
    Returns:
        accuracy (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total


def get_cube_3d_positions(cube_size):
    """
    Get 3D positions for neural cube visualization.
    
    Returns:
        positions: [n_neurons, 3] array of (x, y, z) coordinates
    """
    positions = []
    for z in range(cube_size):
        for y in range(cube_size):
            for x in range(cube_size):
                positions.append([x, y, z])
    return np.array(positions, dtype=np.float32)
