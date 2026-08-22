import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=7):
        super().__init__()
        # Standard Sinusoidal Positional Encoding
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))

    def forward(self, x):
        # x: [B, SeqLen, D]
        # Allow dynamic sequence length inference/cutting
        seq_len = x.size(1)
        # Ensure max_seq_len is sufficient or handle dynamically? 
        # Current implementation relies on max_seq_len being large enough.
        if seq_len > self.pos_enc.size(1):
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.pos_enc.size(1)}")

        return x + self.pos_enc[:, :seq_len, :]

class TransformerModule(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        seq_len=7,
        nhead=4,
        num_layers=4,
        ff_dim_factor=4,
        dropout=0.1,
    ):
        """
        Transformer Encoder that returns BOTH the full sequence (for LFP) 
        and the compressed policy latent (for Action).
        """
        super().__init__()
        # +5 buffer just in case context size varies slightly or off-by-one
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True, # Pre-LN for better stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """
        Args:
            x: [B, SeqLen, Dim]
        Returns:
            x: [B, SeqLen, Dim] Contextualized features
        """
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return x
