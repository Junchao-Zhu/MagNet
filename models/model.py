import numpy as np
from torch.nn import Linear
import torch
from .ResNet import ResNet50, ResNet101, ResNet152
import torch.nn as nn
import torch
from .attention_transformer import GNNTransformerBlock, CrossAttention
from .bs_block import gs_block_with_attention_fixed_weights


class MagNet(nn.Module):
    def __init__(self, encoder_output=1024, num_heads=8, cross_attention_dim=1024, dropout=0.1, transformer_dim=1024,
                 gcn=True, gs_dim=1024, policy='mean', gs_depth=3, gene_output=250):
        super(MagNet, self).__init__()
        self.num_heads = num_heads

        self.encoder = ResNet50(num_classes=encoder_output)
        self.cross_attention_512 = CrossAttention(dim=cross_attention_dim, num_heads=num_heads)
        self.cross_attention_224 = CrossAttention(dim=cross_attention_dim, num_heads=num_heads)

        self.gat_layer_112 = nn.ModuleList(
            [gs_block_with_attention_fixed_weights(gs_dim, gs_dim, policy, gcn) for i in range(gs_depth)])

        self.gat_layer_512 = nn.ModuleList(
            [gs_block_with_attention_fixed_weights(gs_dim, gs_dim, policy, gcn) for i in range(gs_depth)])

        self.gat_layer_224 = nn.ModuleList(
            [gs_block_with_attention_fixed_weights(gs_dim, gs_dim, policy, gcn) for i in range(gs_depth)])

        self.transformer = GNNTransformerBlock(transformer_dim, dropout=dropout, num_heads=num_heads)

        self.lstm_512 = nn.Sequential(
            nn.LSTM(transformer_dim, transformer_dim, 2),
        )

        self.gene_head_250 = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, gene_output)
        )

        self.gene_head_512 = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, gene_output)
        )

        self.gene_head_224 = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, gene_output)
        )

    def forward(self, x, adj, feature_512, feature_224, graph_512, graph_224, adj_512, adj_224):
        x = self.encoder(x)

        x_cross_512 = self.cross_attention_512(x, feature_512, feature_512)
        x_cross_224 = self.cross_attention_224(x, feature_224, feature_224)

        x = x + x_cross_512 + x_cross_224

        graph_x_112 = []

        for layer in self.gat_layer_112:
            g = layer(x, adj)
            graph_x_112.append(g.unsqueeze(0))

        g_112 = torch.cat(graph_x_112, 0)
        g_112 = self.transformer(g_112)

        graph_x_512 = []
        for layer in self.gat_layer_512:
            g = layer(graph_512, adj_512)
            graph_x_512.append(g.unsqueeze(0))

        g_512 = torch.cat(graph_x_512, 0)
        g_512 = self.transformer(g_512)

        graph_x_224 = []
        for layer in self.gat_layer_224:
            g = layer(graph_224, adj_224)
            graph_x_224.append(g.unsqueeze(0))

        g_224 = torch.cat(graph_x_224, 0)
        g_224 = self.transformer(g_224)

        out_112 = self.gene_head_250(g_112)
        out_512 = self.gene_head_250(g_512)
        out_224 = self.gene_head_250(g_224)

        return out_112, out_512, out_224
