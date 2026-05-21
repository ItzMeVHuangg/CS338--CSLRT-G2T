import torch
import torch.nn as nn

class TemporalPool(nn.Module):
    """
    Temporal downsampling for 2D frame features (e.g. from ResNet).
    Reduces sequence length by a factor of 2^num_pool_layers.
    """
    def __init__(self, num_pool_layers: int = 2):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        
        layers = []
        for _ in range(num_pool_layers):
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
        self.pool = nn.Sequential(*layers) if num_pool_layers > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T', D)
        """
        if self.num_pool_layers == 0:
            return x
            
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (B, T', D)
        return x

    def adjust_lengths(self, lens: torch.Tensor, T_in: int = None) -> torch.Tensor:
        """
        Adjust sequence lengths after temporal pooling.
        """
        if self.num_pool_layers == 0:
            return lens
            
        out_lens = lens.float()
        for _ in range(self.num_pool_layers):
            out_lens = torch.floor((out_lens - 2) / 2 + 1)
            
        return out_lens.long().clamp(min=1)
