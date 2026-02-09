"""
Mini-GPT: Transformer-based Language Model
Implements a small-scale GPT architecture for next-token prediction
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask (prevent attending to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x

class MiniGPT(nn.Module):
    """
    Mini-GPT: Small-scale transformer language model
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension (64-256)
        num_heads: Number of attention heads (2-4)
        num_layers: Number of transformer layers (1-2)
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(
        self, 
        vocab_size, 
        embed_dim=128, 
        num_heads=4, 
        num_layers=2,
        max_seq_len=128,
        dropout=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        ff_dim = 4 * embed_dim  # Standard: 4x embedding dimension
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output projection to vocabulary
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass
        
        Args:
            input_ids: Token indices (batch_size, seq_len)
            targets: Target token indices for loss computation (optional)
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_ids.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Positional embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        Generate text autoregressively
        
        Args:
            input_ids: Starting token indices (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            input_ids_cond = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(input_ids_cond)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test instantiation
if __name__ == "__main__":
    # Example configuration
    vocab_size = 50257  # GPT-2 vocabulary size
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=128
    )
    
    print("=" * 60)
    print("Mini-GPT Model")
    print("=" * 60)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Embedding dimension: {model.embed_dim}")
    print(f"Number of layers: {len(model.blocks)}")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(dummy_input, dummy_targets)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print("=" * 60)
