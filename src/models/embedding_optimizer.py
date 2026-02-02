"""
Embedding Optimizer Module

This module projects CLIP image embeddings into the textual embedding space of Stable Diffusion.
It enables image-based conditioning by translating visual features into text-compatible embeddings.

Architecture:
1. Input: CLIP image embeddings [batch_size, 257, 1024]
2. Projection Layer: 1024 → 768 dimensions
3. Attention Pooling: 257 → 77 sequence length
4. MLP Refinement: Four FC layers with ReLU activations
5. Output: Text-compatible embeddings [batch_size, 77, 768]

Reference: Visually Guided Object Insertion Into Image (Tsach & Spira, 2024)
"""

"""
Embedding Optimizer Module

This module projects CLIP image embeddings into the textual embedding space of Stable Diffusion.
It enables image-based conditioning by translating visual features into text-compatible embeddings.

Architecture:
1. Input: CLIP image embeddings [batch_size, 257, 1024]
2. Projection Layer: 1024 → 768 dimensions
3. Attention Pooling: 257 → 77 sequence length
4. MLP Refinement: Four FC layers with ReLU activations
5. Output: Text-compatible embeddings [batch_size, 77, 768]

Reference: Visually Guided Object Insertion Into Image (Tsach & Spira, 2024)
"""

from typing import Tuple

import torch
import torch.nn as nn


class EmbeddingOptimizer(nn.Module):
    """
    Embedding Optimizer for converting CLIP image embeddings to text-compatible embeddings.
    
    This module bridges the gap between CLIP's image embedding space (1024 dim, 257 tokens)
    and Stable Diffusion's text embedding space (768 dim, 77 tokens).
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 768,
        input_seq_len: int = 257,
        output_seq_len: int = 77
    ):
        """
        Initialize the Embedding Optimizer.
        
        Args:
            input_dim: Input embedding dimension (CLIP image encoder output)
            output_dim: Output embedding dimension (Stable Diffusion text encoder input)
            input_seq_len: Input sequence length (CLIP image tokens)
            output_seq_len: Output sequence length (Stable Diffusion text tokens)
        """
        super(EmbeddingOptimizer, self).__init__()
        self.projection_layer = nn.Linear(input_dim, output_dim)
        self.attention_pooling = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        self.mlp_fc1 = nn.Linear(output_dim, output_dim)
        self.mlp_relu = nn.ReLU()
        self.mlp_fc2 = nn.Linear(output_dim, output_dim)
        self.mlp_fc3 = nn.Linear(output_dim, output_dim)
        self.mlp_fc4 = nn.Linear(output_dim, output_dim)

        self.output_seq_len = output_seq_len
        self.input_seq_len = input_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Embedding Optimizer.
        
        Args:
            x: Input CLIP image embeddings of shape [batch_size, 257, 1024]
            
        Returns:
            Output embeddings of shape [batch_size, 77, 768] compatible with Stable Diffusion
        """
        # Step 1: Project input embeddings from 1024 to 768 dimensions
        x = self.projection_layer(x)  # Shape: [batch_size, 257, 768]

        # Step 2: Attention pooling to reduce sequence length from 257 to 77
        # Create a query tensor for pooling: [batch_size, 77, 768]
        query = torch.randn(x.size(0), self.output_seq_len, x.size(-1), device=x.device, dtype=x.dtype)
        x, _ = self.attention_pooling(query, x, x)

        # Step 3: Apply the MLP block for further refinement
        # Four-layer MLP as per final architecture (improved from initial 2-layer design)
        x = self.mlp_fc1(x)
        x = self.mlp_relu(x)
        x = self.mlp_fc2(x)
        x = self.mlp_relu(x)
        x = self.mlp_fc3(x)
        x = self.mlp_relu(x)
        x = self.mlp_fc4(x)

        return x

class AttentionPooling(nn.Module):
    """
    Attention-based pooling module for sequence length reduction.
    
    Uses the CLS token as a query to pool information from the entire sequence.
    """
    
    def __init__(self, input_dim: int, output_tokens: int):
        """
        Initialize Attention Pooling.
        
        Args:
            input_dim: Input embedding dimension
            output_tokens: Desired output sequence length
        """
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.output_tokens = output_tokens

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to reduce sequence length.
        
        Args:
            hidden_states: Input embeddings of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Pooled embeddings of shape [batch_size, output_tokens, input_dim]
        """
        # Compute attention scores using CLS token as query
        query = self.query(hidden_states[:, 0:1, :])  # Use CLS token as the query
        attention_scores = torch.matmul(query, hidden_states.transpose(-1, -2)) / (
            hidden_states.size(-1) ** 0.5
        )
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention pooling
        pooled_output = torch.matmul(attention_weights, hidden_states)  # Shape: [batch_size, 1, input_dim]
        pooled_output = pooled_output.expand(-1, self.output_tokens, -1)  # Expand to match sequence length
        return pooled_output
