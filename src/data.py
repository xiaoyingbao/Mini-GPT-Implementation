"""
Data loading utilities for Mini-GPT training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, concatenate_datasets
import os

class TokenizedDataset(Dataset):
    """
    PyTorch Dataset wrapper for pre-tokenized text data from Assignment 1
    
    Args:
        data_path: Path to tokenized data directory
    """
    def __init__(self, data_path: str):
        print(f"\nLoading tokenized data from: {data_path}")
        
        # Load all tokenized datasets
        dataset_names = ["openwebtext_tokenized", "wikipedia_tokenized", "c4_tokenized"]
        datasets = []
        
        for name in dataset_names:
            path = os.path.join(data_path, name)
            try:
                ds = load_from_disk(path)
                datasets.append(ds)
                print(f"  ✓ Loaded {name}: {len(ds):,} samples")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
        
        # Concatenate all datasets
        self.dataset = concatenate_datasets(datasets)
        print(f"  Total samples: {len(self.dataset):,}\n")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        """
        Get a single training sample
        
        Returns:
            input_ids: Token IDs tensor
            labels: Target token IDs (same as input_ids for language modeling)
        """
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        
        # For causal language modeling, targets are the same as inputs
        # Loss computation will handle the shifting internally
        labels = input_ids.clone()
        
        return input_ids, labels

def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create PyTorch DataLoader for training
    
    Args:
        data_path: Path to tokenized data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU
        
    Returns:
        DataLoader instance
    """
    dataset = TokenizedDataset(data_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    print(f"DataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(dataloader):,}")
    print(f"  Shuffle: {shuffle}\n")
    
    return dataloader

# Test
if __name__ == "__main__":
    # Test data loading
    data_path = "../Assignment1/processed_data/tokenized"
    
    print("=" * 70)
    print("Testing Data Loading")
    print("=" * 70)
    
    dataloader = create_dataloader(
        data_path=data_path,
        batch_size=32,
        shuffle=True
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    input_ids, labels = batch
    
    print("Sample batch:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  First sequence (10 tokens): {input_ids[0][:10].tolist()}")
    print("=" * 70)
