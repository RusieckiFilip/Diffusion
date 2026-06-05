"""
model.py
--------
U-Net noise-prediction network for the DDPM backward process.

Architecture mirrors the one from the YouTube tutorial
(https://www.youtube.com/watch?v=a4Yfz2FxXiY) with two small improvements:
  1. GroupNorm instead of BatchNorm — more stable at small batch sizes and
     works correctly during single-image inference (sampling).
  2. Slightly larger time_emb_dim (256) so the network can distinguish
     timesteps more easily across T=300 steps.

Parameter count: ~62 M  (same ballpark as the tutorial).
"""

import math
import torch
import torch.nn as nn


# =========================================================================== #
# Sinusoidal Position Embeddings (time → vector)                              #
# =========================================================================== #

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes a scalar timestep t into a vector of length *dim* using
    sinusoidal positional encodings (same as in 'Attention Is All You Need').
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device   = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]          # (B, D/2)
        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()), dim=-1
        )                                                         # (B, D)
        return embeddings


# =========================================================================== #
# Residual convolutional block with time injection                             #
# =========================================================================== #

class Block(nn.Module):
    """
    One encoder or decoder block of the U-Net.

    Down block  (up=False):  conv → downsample
    Up   block  (up=True ):  conv → upsample (transposed conv)

    The timestep embedding is projected to *out_ch* and added to the
    feature maps after the first convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 up: bool = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1     = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # GroupNorm: stable for any batch size, including batch=1 at inference
        self.gnorm1 = nn.GroupNorm(8, out_ch)
        self.gnorm2 = nn.GroupNorm(8, out_ch)
        self.act    = nn.SiLU()   # Swish — slightly smoother than ReLU

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # --- First conv ---
        h = self.gnorm1(self.act(self.conv1(x)))
        # --- Inject time embedding (broadcast over H, W) ---
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[..., None, None]        # (B, C, 1, 1)
        h = h + time_emb
        # --- Second conv ---
        h = self.gnorm2(self.act(self.conv2(h)))
        # --- Down / up sample ---
        return self.transform(h)


# =========================================================================== #
# Simple U-Net                                                                 #
# =========================================================================== #

class SimpleUnet(nn.Module):
    """
    Simplified U-Net that predicts the noise ε added at timestep t.

    Input  : (B, 3, H, W)  — noisy image  +  (B,) — timestep indices
    Output : (B, 3, H, W)  — predicted noise

    The skip connections concatenate encoder feature maps with the
    corresponding decoder feature maps (standard U-Net).
    """

    def __init__(
        self,
        image_channels: int = 3,
        down_channels: tuple = (64, 128, 256, 512, 1024),
        up_channels:   tuple = (1024, 512, 256, 128, 64),
        out_dim:       int   = 3,
        time_emb_dim:  int   = 256,
    ):
        super().__init__()

        # ---- Time embedding MLP ------------------------------------------ #
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # ---- Initial projection ------------------------------------------ #
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # ---- Encoder (downsample) ----------------------------------------- #
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])

        # ---- Decoder (upsample) ------------------------------------------ #
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        # ---- Final 1×1 projection ---------------------------------------- #
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Embed time
        t = self.time_mlp(timestep)                 # (B, time_emb_dim)

        # Initial conv
        x = self.conv0(x)                           # (B, 64, H, W)

        # Encoder — save residuals for skip connections
        residual_inputs: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        # Decoder — concatenate skip connections
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)  # channel-wise concat
            x = up(x, t)

        return self.output(x)


# =========================================================================== #
# Quick sanity check                                                           #
# =========================================================================== #
if __name__ == "__main__":
    model = SimpleUnet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleUnet  |  parameters: {n_params:,}")

    # Forward pass
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 300, (2,)).long()
    out = model(x, t)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch!"
    print("Shape check passed ✓")
