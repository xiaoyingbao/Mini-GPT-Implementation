"""
Training script for Mini-GPT
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from src.model import MiniGPT
from src.data import create_dataloader
from src.utils import (
    load_config, 
    save_config,
    calculate_perplexity,
    count_parameters,
    save_checkpoint,
    save_training_history,
    AverageMeter,
    format_time
)
import time

def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    device,
    epoch: int
):
    """
    Train for one epoch
    
    Returns:
        avg_loss, avg_perplexity
    """
    model.train()
    loss_meter = AverageMeter()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (input_ids, targets) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits, loss = model(input_ids, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), input_ids.size(0))
        
        # Update progress bar
        perplexity = calculate_perplexity(loss_meter.avg)
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'ppl': f'{perplexity:.2f}'
        })
    
    avg_loss = loss_meter.avg
    avg_perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, avg_perplexity

def train(config_path: str):
    """
    Main training function
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    print("=" * 80)
    print("MINI-GPT TRAINING")
    print("=" * 80)
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Create directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Create experiment subdirectory
    exp_name = config['experiment']['name']
    exp_checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], exp_name)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    
    # Save config to experiment directory
    config_save_path = os.path.join(exp_checkpoint_dir, 'config.yaml')
    save_config(config, config_save_path)
    
    # Create dataloader
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    dataloader = create_dataloader(
        data_path=config['data']['train_data_path'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    # Initialize model
    print("=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    model = MiniGPT(
        vocab_size=config['model']['vocab_size'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout']
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Embedding dimension: {config['model']['embed_dim']}")
    print(f"  Number of attention heads: {config['model']['num_heads']}")
    print(f"  Number of layers: {config['model']['num_layers']}")
    print(f"  Max sequence length: {config['model']['max_seq_len']}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    print(f"Optimizer: AdamW")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}\n")
    
    # Training history
    history = {
        'epochs': [],
        'losses': [],
        'perplexities': [],
        'time_per_epoch': []
    }
    
    # Training loop
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    total_start_time = time.time()
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        epoch_start_time = time.time()
        
        # Train one epoch
        avg_loss, avg_perplexity = train_epoch(
            model, dataloader, optimizer, device, epoch
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['epochs'].append(epoch)
        history['losses'].append(avg_loss)
        history['perplexities'].append(avg_perplexity)
        history['time_per_epoch'].append(epoch_time)
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']} Summary")
        print(f"{'='*80}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {avg_perplexity:.2f}")
        print(f"  Time: {format_time(epoch_time)}")
        print(f"{'='*80}\n")
        
        # Save checkpoint
        if epoch % config['logging']['save_every_epoch'] == 0:
            checkpoint_path = os.path.join(
                exp_checkpoint_dir,
                f"checkpoint_epoch{epoch}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch, avg_loss, avg_perplexity,
                config, checkpoint_path
            )
        
        # Save training history
        history_path = os.path.join(exp_checkpoint_dir, 'training_history.json')
        save_training_history(history, history_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        exp_checkpoint_dir,
        "mini_gpt_final.pt"
    )
    save_checkpoint(
        model, optimizer, 
        config['training']['num_epochs'],
        history['losses'][-1],
        history['perplexities'][-1],
        config, 
        final_checkpoint_path
    )
    
    total_time = time.time() - total_start_time
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nExperiment: {exp_name}")
    print(f"\nFinal Metrics:")
    print(f"  Final Loss: {history['losses'][-1]:.4f}")
    print(f"  Final Perplexity: {history['perplexities'][-1]:.2f}")
    print(f"  Total Training Time: {format_time(total_time)}")
    print(f"  Average Time per Epoch: {format_time(sum(history['time_per_epoch'])/len(history['time_per_epoch']))}")
    print(f"\nCheckpoints saved in: {exp_checkpoint_dir}")
    print("=" * 80)
    
    return model, history

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Mini-GPT')
    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, history = train(args.config)

if __name__ == "__main__":
    main()
