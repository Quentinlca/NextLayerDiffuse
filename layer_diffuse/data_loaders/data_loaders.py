import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
# Constants
from constants import *




class DatasetWithPrompts(torch.utils.data.Dataset):
        def __init__(self, dataset, prompts):
            self.dataset = dataset
            self.prompts = np.array(prompts)
            assert len(self.dataset) == len(self.prompts), "Dataset and prompts must be the same length"

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            X, Y = self.dataset[idx]
            prompt = self.prompts[idx]
            return X, Y, prompt

def get_loaders(dataset_path, shuffle=True, batch_size=32, num_workers=4, train_size=0.8):
    """
    Get DataLoader for the dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        shuffle (bool): Whether to shuffle the dataset. Default: True
        batch_size (int): Batch size for DataLoader. Default: 32
        num_workers (int): Number of workers for DataLoader. Default: 4

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    from torchvision import datasets, transforms

    dataset = torch.load(dataset_path,weights_only=False)
    #split the dataset into train and validation sets
    train_size = int(len(dataset) * train_size)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Create DataLoader for training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # Return both loaders
    return train_loader, val_loader

def get_loaders_with_prompts(dataset_path, prompts_path, shuffle=True, batch_size=32, num_workers=4, train_size=0.8):
    """
    Get DataLoader for the dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        shuffle (bool): Whether to shuffle the dataset. Default: True
        batch_size (int): Batch size for DataLoader. Default: 32
        num_workers (int): Number of workers for DataLoader. Default: 4

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    from torchvision import datasets, transforms

    dataset = torch.load(dataset_path,weights_only=False)
    prompts = pd.read_csv(prompts_path)['prompts'].to_list()
    # Combine dataset and prompts into a custom Dataset
    
    dataset = DatasetWithPrompts(dataset, prompts)
    #split the dataset into train and validation sets
    train_size = int(len(dataset) * train_size)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Create DataLoader for training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # Return both loaders
    return train_loader, val_loader