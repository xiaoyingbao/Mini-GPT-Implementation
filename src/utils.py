"""
Utility functions for training and evaluation
"""

import yaml
import json
import math
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss
    
    Perplexity = exp(loss)
    """
    return math.exp(loss)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    perplexity: float,
    config: Dict[str, Any],
    filepath: str
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Training loss
        perplexity: Perplexity score
        config: Training configuration
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    print(f"  ✓ Checkpoint saved: {filepath}")

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Perplexity: {checkpoint['perplexity']:.2f}")
    
    return checkpoint

def save_training_history(history: Dict[str, list], filepath: str):
    """
    Save training history to JSON
    
    Args:
        history: Dictionary with training metrics
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Training history saved: {filepath}")

def load_training_history(filepath: str) -> Dict[str, list]:
    """Load training history from JSON"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def format_time(seconds: float) -> str:
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Utilities")
    print("=" * 70)
    
    # Test config loading
    config = load_config("config/base_config.yaml")
    print("\n✓ Config loaded:")
    print(f"  Model embed_dim: {config['model']['embed_dim']}")
    print(f"  Training batch_size: {config['training']['batch_size']}")
    
    # Test perplexity calculation
    loss = 2.5
    ppl = calculate_perplexity(loss)
    print(f"\n✓ Perplexity calculation:")
    print(f"  Loss: {loss:.4f} → Perplexity: {ppl:.2f}")
    
    # Test time formatting
    print(f"\n✓ Time formatting:")
    print(f"  3665 seconds → {format_time(3665)}")
    
    print("=" * 70)
