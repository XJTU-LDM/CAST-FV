"""Coordinate and boundary channels used by CAST-FV state maps."""

from __future__ import annotations

import math

import torch

from .config import CaseConfig


def coordinate_boundary_features(case: CaseConfig, device: torch.device | str) -> torch.Tensor:
    """Return cell-centre coordinates, smooth coordinates and wall indicators.

    Two-dimensional tensors use ``[batch, channel, y, x]`` ordering. Three-
    dimensional tensors use ``[batch, channel, z, y, x]`` ordering, with the
    moving/heated lid at ``z = 1``.
    """

    device = torch.device(device)
    line = (torch.arange(case.cells, device=device, dtype=torch.float32) + 0.5) / case.cells
    index = torch.arange(case.cells, device=device)

    if case.dimension == 2:
        yy, xx = torch.meshgrid(line, line, indexing="ij")
        top = (index[:, None] == case.cells - 1).float().expand(case.cells, case.cells)
        bottom = (index[:, None] == 0).float().expand(case.cells, case.cells)
        left = (index[None, :] == 0).float().expand(case.cells, case.cells)
        right = (index[None, :] == case.cells - 1).float().expand(case.cells, case.cells)
        features = torch.stack(
            [xx, yy, torch.sin(math.pi * xx), torch.sin(math.pi * yy), top, bottom, left, right],
            dim=0,
        )
        return features.unsqueeze(0)

    zz, yy, xx = torch.meshgrid(line, line, line, indexing="ij")
    top = (index[:, None, None] == case.cells - 1).float().expand(*case.spatial_shape)
    bottom = (index[:, None, None] == 0).float().expand(*case.spatial_shape)
    x_min = (index[None, None, :] == 0).float().expand(*case.spatial_shape)
    x_max = (index[None, None, :] == case.cells - 1).float().expand(*case.spatial_shape)
    y_min = (index[None, :, None] == 0).float().expand(*case.spatial_shape)
    y_max = (index[None, :, None] == case.cells - 1).float().expand(*case.spatial_shape)
    features = torch.stack(
        [
            xx,
            yy,
            zz,
            torch.sin(math.pi * xx),
            torch.sin(math.pi * yy),
            torch.sin(math.pi * zz),
            top,
            bottom,
            x_min,
            x_max,
            y_min,
            y_max,
        ],
        dim=0,
    )
    return features.unsqueeze(0)
