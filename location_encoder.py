import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (
        2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)
    ) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


class LocationEncoder(nn.Module):
    def __init__(
        self,
        sigma=[2**0, 2**4, 2**8, 2**12],
        embed_dim=768,
        n_registers=4,
        num_heads=12,
        mlp_ratio=4,
        depth=12,
    ):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.n_registers = n_registers

        self.loc_enc = nn.ModuleList(
            [
                GaussianEncoding(sigma=sigma, input_size=2, encoded_size=embed_dim // 2)
                for sigma in self.sigma
            ]
        )

        self.registers = nn.Parameter(torch.empty(n_registers, embed_dim))
        nn.init.normal_(self.registers, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = []

        for gaussian_encoding in self.loc_enc:
            location_features.append(gaussian_encoding(location))

        location_features = torch.stack(location_features, dim=1)
        registers = self.registers.unsqueeze(0).expand(
            location_features.shape[0], -1, -1
        )
        tokens = torch.cat([registers, location_features], dim=1)

        tokens = self.transformer(tokens)

        return tokens[:, self.n_registers :, :].mean(dim=1)


class GaussianEncoding(nn.Module):
    """Map coordinates to a Gaussian Random Fourier Features embedding."""

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[int] = None,
        encoded_size: Optional[int] = None,
    ):
        super().__init__()
        b = torch.randn(int(encoded_size), int(input_size)) * float(sigma)
        self.mat = nn.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        vp = 2 * np.pi * v @ self.mat.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

