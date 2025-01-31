import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=3, dropout=0.1):
        """
        dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers
        dropout: Dropout probability
        """
        super(GNNTransformerBlock, self).__init__()

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Learnable positional encoding to distinguish hierarchical order
        self.positional_encoding = nn.Parameter(torch.randn(num_layers, dim))

    def forward(self, gnn_outputs):
        """
        gnn_outputs: Multi-layer GNN output in the format (batch_size, num_layers, num_nodes, dim)
        """
        # Reshape input, treating num_nodes and num_layers as sequence length
        num_layers, num_nodes, dim = gnn_outputs.size()
        batch_size = 1
        gnn_outputs = gnn_outputs.view(batch_size * num_nodes, num_layers, dim)

        gnn_outputs += self.positional_encoding.unsqueeze(0).repeat(batch_size * num_nodes, 1, 1)
        transformer_output = self.transformer_encoder(gnn_outputs)  # Output shape: (batch_size * num_nodes, num_layers, dim)

        # Aggregate layer outputs, e.g., take the output of the last layer or the average of all layers
        final_output = transformer_output.mean(dim=1)  # (batch_size * num_nodes, dim)

        final_output = final_output.view(num_nodes, dim)
        return final_output


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "Embedding dimension must be divisible by number of heads"

        # Define linear transformations for Query, Key, and Value
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # Output linear layer
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        The shape of query, key, and value is [batch_size, feature_size]
        """
        batch_size, feature_size = query.size()

        # Treat [batch_size, feature_size] as [batch_size, 1, feature_size] by adding a seq_len dimension
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # Linear transformation: project to multi-head dimensions
        query = self.query_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=query.device))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)  # Apply Dropout

        # Use attention scores to weight Value
        context = torch.matmul(attention_probs, value)

        # Merge multi-head outputs
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)

        # Final linear layer to map back to the original dimension
        output = self.out_proj(context).squeeze(1)
        return output
