import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random

class HSIDataLoader:
    def __init__(self, dataset_name, data_dir='Final/Data', patch_size=9, 
                 batch_size=32, train_split=0.2, val_split=0.6,
                 random_seed=42, num_workers=4):
        """
        Initialize HSI Data Loader
        
        Args:
            dataset_name (str): Name of the dataset (e.g., 'Indian_pines', 'Pavia')
            data_dir (str): Directory containing the .mat files
            patch_size (int): Size of spatial patches
            batch_size (int): Batch size for DataLoader
            train_split (float): Proportion of data for training
            val_split (float): Proportion of data for validation
            random_seed (int): Random seed for reproducibility
            num_workers (int): Number of workers for DataLoader
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.random_seed = random_seed
        self.num_workers = num_workers
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Load and process data
        self.data, self.labels = self._load_data()
        self.num_classes = len(np.unique(self.labels))
        self.num_channels = self.data.shape[-1]
        
    def _find_mat_files(self):
        """Find corresponding data and ground truth files"""
        files = os.listdir(self.data_dir)
        
        # Find data file
        data_pattern = re.compile(f".*{self.dataset_name}.*(?<!_gt)\.mat$", re.IGNORECASE)
        data_files = [f for f in files if data_pattern.match(f)]
        
        # Find ground truth file
        gt_pattern = re.compile(f".*{self.dataset_name}.*_gt\.mat$", re.IGNORECASE)
        gt_files = [f for f in files if gt_pattern.match(f)]
        
        if not data_files or not gt_files:
            raise FileNotFoundError(f"Could not find matching .mat files for {self.dataset_name}")
            
        return os.path.join(self.data_dir, data_files[0]), os.path.join(self.data_dir, gt_files[0])
    
    def _load_data(self):
        """Load and preprocess the data"""
        # Find data files
        data_file, gt_file = self._find_mat_files()
        
        # Load .mat files
        try:
            data = sio.loadmat(data_file)
            labels = sio.loadmat(gt_file)
            
            # Extract arrays (handle different key naming conventions)
            data_key = [k for k in data.keys() if not k.startswith('__')][0]
            gt_key = [k for k in labels.keys() if not k.startswith('__')][0]
            
            data = data[data_key].astype(np.float32)
            labels = labels[gt_key].astype(np.int64)
            
            # Normalize the data
            shaped_data = data.reshape(-1, data.shape[-1])
            scaler = StandardScaler()
            shaped_data = scaler.fit_transform(shaped_data)
            data = shaped_data.reshape(data.shape)
            
            return data, labels
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _create_patch_dataset(self, data, labels, train=True):
        """Create a patch-based dataset"""
        class PatchDataset(Dataset):
            def __init__(self, data, labels, patch_size, train=True):
                self.data = torch.FloatTensor(data)
                self.labels = torch.LongTensor(labels)
                self.patch_size = patch_size
                self.train = train
                
                # Calculate padding
                self.pad_size = patch_size // 2
                
                # Pad data
                self.padded_data = F.pad(
                    self.data.permute(2, 0, 1),
                    (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                    mode='reflect'
                ).permute(1, 2, 0)
                
                # Pad labels
                self.padded_labels = F.pad(
                    self.labels,
                    (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                    mode='constant',
                    value=0
                )
                
                # Get valid positions (non-zero labels)
                self.valid_positions = [
                    (i, j) for i in range(labels.shape[0])
                    for j in range(labels.shape[1])
                    if labels[i, j] != 0
                ]
            
            def __len__(self):
                return len(self.valid_positions)
            
            def _augment(self, patch, label_patch):
                """Apply random augmentations"""
                # Random rotation
                k = random.randint(0, 3)
                if k > 0:
                    patch = torch.rot90(patch, k, dims=[-2, -1])
                    label_patch = torch.rot90(label_patch, k, dims=[-2, -1])
                
                # Random flip
                if random.random() > 0.5:
                    patch = torch.flip(patch, dims=[-2])
                    label_patch = torch.flip(label_patch, dims=[-2])
                if random.random() > 0.5:
                    patch = torch.flip(patch, dims=[-1])
                    label_patch = torch.flip(label_patch, dims=[-1])
                
                return patch, label_patch
            
            def __getitem__(self, idx):
                # Get center position
                i, j = self.valid_positions[idx]
                i += self.pad_size
                j += self.pad_size
                
                # Extract patches
                patch = self.padded_data[
                    i-self.pad_size:i+self.pad_size+1,
                    j-self.pad_size:j+self.pad_size+1,
                    :
                ]
                label_patch = self.padded_labels[
                    i-self.pad_size:i+self.pad_size+1,
                    j-self.pad_size:j+self.pad_size+1
                ]
                
                # Convert patch to (C, H, W) format
                patch = patch.permute(2, 0, 1)
                
                # Apply augmentation for training
                if self.train and random.random() > 0.5:
                    patch, label_patch = self._augment(patch, label_patch)
                
                return patch, label_patch
        
        return PatchDataset(data, labels, self.patch_size, train)
    
    def _split_data(self):
        """Split data into train, validation, and test sets with stratification"""
        height, width = self.labels.shape
        # Create list of (position, class) tuples for non-zero labels
        labeled_pixels = [(i, j, self.labels[i, j]) 
                        for i in range(height) 
                        for j in range(width) 
                        if self.labels[i, j] != 0]
        
        # Group indices by class
        class_indices = {}
        for i, j, label in labeled_pixels:
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append((i, j))
        
        # Split each class proportionally
        train_idx, val_idx, test_idx = [], [], []
        for class_label, indices in class_indices.items():
            n_samples = len(indices)
            n_train = int(n_samples * self.train_split)
            n_val = int(n_samples * self.val_split)
            
            # Shuffle indices
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            
            # Split
            train_idx.extend(shuffled_indices[:n_train])
            val_idx.extend(shuffled_indices[n_train:n_train + n_val])
            test_idx.extend(shuffled_indices[n_train + n_val:])
        
        # Create masks
        train_mask = np.zeros_like(self.labels)
        val_mask = np.zeros_like(self.labels)
        test_mask = np.zeros_like(self.labels)
        
        for i, j in train_idx:
            train_mask[i, j] = self.labels[i, j]
        for i, j in val_idx:
            val_mask[i, j] = self.labels[i, j]
        for i, j in test_idx:
            test_mask[i, j] = self.labels[i, j]
        
        # Print class distribution
        def get_class_distribution(mask):
            unique, counts = np.unique(mask, return_counts=True)
            return dict(zip(unique, counts))
        
        print("\nClass Distribution (excluding background class 0):")
        print("\nTraining Set:")
        train_dist = get_class_distribution(train_mask)
        for class_label in sorted(train_dist.keys()):
            if class_label != 0:  # Skip background class
                print(f"Class {class_label}: {train_dist[class_label]} samples")
        
        print("\nValidation Set:")
        val_dist = get_class_distribution(val_mask)
        for class_label in sorted(val_dist.keys()):
            if class_label != 0:
                print(f"Class {class_label}: {val_dist[class_label]} samples")
        
        print("\nTest Set:")
        test_dist = get_class_distribution(test_mask)
        for class_label in sorted(test_dist.keys()):
            if class_label != 0:
                print(f"Class {class_label}: {test_dist[class_label]} samples")
        
        # Print split percentages for each class
        print("\nActual Split Percentages per Class:")
        for class_label in sorted(class_indices.keys()):
            total = (train_dist.get(class_label, 0) + 
                    val_dist.get(class_label, 0) + 
                    test_dist.get(class_label, 0))
            train_pct = train_dist.get(class_label, 0) / total * 100
            val_pct = val_dist.get(class_label, 0) / total * 100
            test_pct = test_dist.get(class_label, 0) / total * 100
            print(f"\nClass {class_label}:")
            print(f"Train: {train_pct:.1f}%")
            print(f"Val: {val_pct:.1f}%")
            print(f"Test: {test_pct:.1f}%")
        
        return train_mask, val_mask, test_mask
    
    def get_loaders(self):
        """Get train, validation, and test data loaders"""
        # Split data
        train_mask, val_mask, test_mask = self._split_data()
        
        # Create datasets
        train_dataset = self._create_patch_dataset(self.data, train_mask, train=True)
        val_dataset = self._create_patch_dataset(self.data, val_mask, train=False)
        test_dataset = self._create_patch_dataset(self.data, test_mask, train=False)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'name': self.dataset_name,
            'num_classes': self.num_classes,
            'num_channels': self.num_channels,
            'spatial_size': self.data.shape[:2],
            'patch_size': self.patch_size
        }