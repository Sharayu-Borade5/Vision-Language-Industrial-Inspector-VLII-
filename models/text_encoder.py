import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8),
            num_layers=4
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = self.encoder(x)
        return x.mean(dim=0)
