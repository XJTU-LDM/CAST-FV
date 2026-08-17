"""Neural parameterizations of the complete cell-centred finite-volume state."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CaseConfig, OptimizationConfig


class CNNStateMap(nn.Module):
    """Compact local convolutional state map for two or three dimensions."""

    def __init__(self, case: CaseConfig, width: int | None = None, depth: int = 5):
        super().__init__()
        width = width or (48 if case.dimension == 2 else 32)
        convolution = nn.Conv2d if case.dimension == 2 else nn.Conv3d
        layers: list[nn.Module] = [
            convolution(case.feature_channels, width, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth - 2):
            layers.extend([convolution(width, width, 3, padding=1), nn.GELU()])
        layers.append(convolution(width, case.state_channels, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class UNetStateMap2D(nn.Module):
    """Two-scale state map with a skip connection."""

    def __init__(self, in_channels: int = 8, width: int = 32, out_channels: int = 4):
        super().__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.GELU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(width, out_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fine = self.encoder_1(features)
        coarse = self.encoder_2(F.avg_pool2d(fine, 2))
        coarse = self.middle(coarse)
        upsampled = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.decoder(torch.cat([upsampled, fine], dim=1)))


class TransformerStateMap2D(nn.Module):
    """Global token communication over a two-dimensional cell grid."""

    def __init__(
        self,
        in_channels: int = 8,
        width: int = 96,
        layers: int = 3,
        heads: int = 4,
        out_channels: int = 4,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by the number of heads")
        self.input_projection = nn.Linear(in_channels, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_projection = nn.Linear(width, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        encoded = self.encoder(self.input_projection(tokens))
        return self.output_projection(encoded).reshape(batch, height, width, -1).permute(0, 3, 1, 2)


class GridGNNStateMap2D(nn.Module):
    """Four-neighbour message passing on a Cartesian cell graph."""

    def __init__(self, cells: int, in_channels: int = 8, width: int = 64, steps: int = 5):
        super().__init__()
        self.cells = cells
        self.steps = steps
        self.node_projection = nn.Linear(in_channels, width)
        self.message = nn.Sequential(
            nn.Linear(2 * width + 2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.output = nn.Linear(width, 4)

        edges: list[tuple[int, int]] = []
        offsets: list[tuple[float, float]] = []
        scale = max(1, cells - 1)
        for row in range(cells):
            for column in range(cells):
                source = row * cells + column
                for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbour_row = row + drow
                    neighbour_column = column + dcolumn
                    if 0 <= neighbour_row < cells and 0 <= neighbour_column < cells:
                        edges.append((source, neighbour_row * cells + neighbour_column))
                        offsets.append((dcolumn / scale, drow / scale))
        self.register_buffer("edge_index", torch.tensor(edges, dtype=torch.long), persistent=False)
        self.register_buffer("relative_offset", torch.tensor(offsets, dtype=torch.float32), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        if height != self.cells or width != self.cells:
            raise ValueError(f"expected a {self.cells} x {self.cells} grid")
        nodes = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden = self.node_projection(nodes)
        source, destination = self.edge_index[:, 0], self.edge_index[:, 1]
        relative = self.relative_offset.unsqueeze(0).expand(batch, -1, -1)

        for _ in range(self.steps):
            message_input = torch.cat([hidden[:, source], hidden[:, destination], relative], dim=-1)
            messages = self.message(message_input)
            aggregate = torch.zeros_like(hidden)
            aggregate.scatter_add_(
                1,
                destination.view(1, -1, 1).expand(batch, -1, hidden.shape[-1]),
                messages,
            )
            degree = torch.zeros(batch, height * width, 1, device=features.device, dtype=hidden.dtype)
            degree.scatter_add_(
                1,
                destination.view(1, -1, 1).expand(batch, -1, 1),
                torch.ones(batch, len(destination), 1, device=features.device, dtype=hidden.dtype),
            )
            aggregate = aggregate / degree.clamp_min(1.0)
            hidden = hidden + 0.5 * self.update(torch.cat([hidden, aggregate], dim=-1))

        return self.output(hidden).reshape(batch, height, width, 4).permute(0, 3, 1, 2)


class ResidualCorrector2D(nn.Module):
    """Repeated state correction conditioned on the current FV residual."""

    def __init__(self, feature_channels: int = 8, width: int = 48, steps: int = 4):
        super().__init__()
        self.steps = steps
        input_channels = 4 + 4 + feature_channels + 1
        self.corrector = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 4, 1),
        )
        self.step_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        features: torch.Tensor,
        residual_function: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        state = torch.zeros(
            features.shape[0],
            4,
            features.shape[-2],
            features.shape[-1],
            device=features.device,
            dtype=features.dtype,
        )
        for index in range(self.steps):
            position = torch.full(
                (features.shape[0], 1, features.shape[-2], features.shape[-1]),
                (index + 1) / self.steps,
                device=features.device,
                dtype=features.dtype,
            )
            scaled_residual = torch.tanh(0.05 * residual_function(state)).detach()
            update = self.corrector(torch.cat([state, scaled_residual, features, position], dim=1))
            state = state + self.step_scale.tanh() * update / self.steps
        return state


def build_state_map(case: CaseConfig, optimization: OptimizationConfig) -> nn.Module:
    """Build one fresh, case-specific state map."""

    name = optimization.architecture.lower().replace("-", "_")
    if name == "cnn":
        return CNNStateMap(case, width=optimization.width)
    if case.dimension != 2:
        raise ValueError("the minimal 3D release currently exposes the compact CNN state map only")
    if name == "unet":
        return UNetStateMap2D(width=optimization.width or 32)
    if name == "transformer":
        return TransformerStateMap2D(width=optimization.width or 96)
    if name in ("grid_gnn", "gnn"):
        return GridGNNStateMap2D(case.cells, width=optimization.width or 64)
    if name in ("residual_corrector", "corrector"):
        return ResidualCorrector2D(
            feature_channels=case.feature_channels,
            width=optimization.width or 48,
            steps=optimization.corrector_steps,
        )
    raise ValueError(f"unknown architecture: {optimization.architecture}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
