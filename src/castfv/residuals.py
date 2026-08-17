"""Differentiable Cartesian finite-volume residuals used as CAST-FV authority."""

from __future__ import annotations

import torch

from .config import CaseConfig


def zero_state(case: CaseConfig, device: torch.device | str = "cpu") -> torch.Tensor:
    """Create a quiescent cell-centred interior state."""

    return torch.zeros(
        (1, case.state_channels, *case.spatial_shape),
        device=torch.device(device),
        dtype=torch.float32,
    )


def _neighbour(q: torch.Tensor, axis: int, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    neighbour = torch.roll(q, shifts=shift, dims=axis)
    mask = torch.zeros_like(q, dtype=torch.bool)
    selector = [slice(None)] * q.ndim
    selector[axis] = 0 if shift > 0 else -1
    mask[tuple(selector)] = True
    return torch.where(mask, q, neighbour), mask


def _normal_gradient(
    q: torch.Tensor,
    face_value: torch.Tensor,
    boundary_mask: torch.Tensor,
    axis: int,
    shift: int,
    spacing: float,
) -> torch.Tensor:
    neighbour, _ = _neighbour(q, axis, shift)
    internal = (neighbour - q) / spacing
    boundary = (face_value - q) / (0.5 * spacing)
    return torch.where(boundary_mask, boundary, internal)


def _residual_2d(
    state: torch.Tensor,
    case: CaseConfig,
    previous_state: torch.Tensor | None,
    time_step: float | None,
) -> torch.Tensor:
    pressure, velocity_x, velocity_y, scalar = (
        state[:, 0:1],
        state[:, 1:2],
        state[:, 2:3],
        state[:, 3:4],
    )
    face_specs = {
        "west": (3, 1),
        "east": (3, -1),
        "south": (2, 1),
        "north": (2, -1),
    }
    faces: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, (axis, shift) in face_specs.items():
        interpolated: list[torch.Tensor] = []
        boundary_mask: torch.Tensor | None = None
        for quantity in (pressure, velocity_x, velocity_y, scalar):
            neighbour, mask = _neighbour(quantity, axis, shift)
            boundary_mask = mask
            interpolated.append(0.5 * (quantity + neighbour))
        assert boundary_mask is not None
        pressure_face, velocity_x_face, velocity_y_face, scalar_face = interpolated
        prescribed_x = torch.zeros_like(velocity_x_face)
        prescribed_y = torch.zeros_like(velocity_y_face)
        prescribed_scalar = torch.full_like(scalar_face, case.wall_scalar)
        if name == "north":
            prescribed_x = torch.full_like(velocity_x_face, case.lid_velocity)
            prescribed_scalar = torch.full_like(scalar_face, case.top_scalar)
        faces[name] = (
            torch.where(boundary_mask, pressure, pressure_face),
            torch.where(boundary_mask, prescribed_x, velocity_x_face),
            torch.where(boundary_mask, prescribed_y, velocity_y_face),
            torch.where(boundary_mask, prescribed_scalar, scalar_face),
            boundary_mask,
            axis,
            shift,
        )

    p_w, u_w, v_w, t_w, m_w, a_w, s_w = faces["west"]
    p_e, u_e, v_e, t_e, m_e, a_e, s_e = faces["east"]
    p_s, u_s, v_s, t_s, m_s, a_s, s_s = faces["south"]
    p_n, u_n, v_n, t_n, m_n, a_n, s_n = faces["north"]

    area = case.face_area
    flux_w = -area * u_w
    flux_e = area * u_e
    flux_s = -area * v_s
    flux_n = area * v_n
    fluxes = (flux_w, flux_e, flux_s, flux_n)

    continuity = sum(fluxes)
    momentum_x = case.density * sum(
        flux * face for flux, face in zip(fluxes, (u_w, u_e, u_s, u_n), strict=True)
    )
    momentum_y = case.density * sum(
        flux * face for flux, face in zip(fluxes, (v_w, v_e, v_s, v_n), strict=True)
    )
    scalar_residual = sum(
        flux * face for flux, face in zip(fluxes, (t_w, t_e, t_s, t_n), strict=True)
    )

    momentum_x = momentum_x + area * (p_e - p_w)
    momentum_y = momentum_y + area * (p_n - p_s)

    specifications = (
        (u_w, m_w, a_w, s_w),
        (u_e, m_e, a_e, s_e),
        (u_s, m_s, a_s, s_s),
        (u_n, m_n, a_n, s_n),
    )
    diffusion_x = sum(
        _normal_gradient(velocity_x, face, mask, axis, shift, case.spacing)
        for face, mask, axis, shift in specifications
    )
    specifications = (
        (v_w, m_w, a_w, s_w),
        (v_e, m_e, a_e, s_e),
        (v_s, m_s, a_s, s_s),
        (v_n, m_n, a_n, s_n),
    )
    diffusion_y = sum(
        _normal_gradient(velocity_y, face, mask, axis, shift, case.spacing)
        for face, mask, axis, shift in specifications
    )
    specifications = (
        (t_w, m_w, a_w, s_w),
        (t_e, m_e, a_e, s_e),
        (t_s, m_s, a_s, s_s),
        (t_n, m_n, a_n, s_n),
    )
    diffusion_scalar = sum(
        _normal_gradient(scalar, face, mask, axis, shift, case.spacing)
        for face, mask, axis, shift in specifications
    )
    momentum_x = momentum_x - case.viscosity * area * diffusion_x
    momentum_y = momentum_y - case.viscosity * area * diffusion_y
    scalar_residual = scalar_residual - (1.0 / case.peclet) * area * diffusion_scalar

    if previous_state is not None:
        if time_step is None or time_step <= 0.0:
            raise ValueError("a positive time_step is required with previous_state")
        momentum_x = momentum_x + case.density * case.cell_volume * (
            velocity_x - previous_state[:, 1:2]
        ) / time_step
        momentum_y = momentum_y + case.density * case.cell_volume * (
            velocity_y - previous_state[:, 2:3]
        ) / time_step
        scalar_residual = scalar_residual + case.cell_volume * (
            scalar - previous_state[:, 3:4]
        ) / time_step

    return torch.cat([continuity, momentum_x, momentum_y, scalar_residual], dim=1) / case.cell_volume


def _residual_3d(
    state: torch.Tensor,
    case: CaseConfig,
    previous_state: torch.Tensor | None,
    time_step: float | None,
) -> torch.Tensor:
    pressure, velocity_x, velocity_y, velocity_z, scalar = (
        state[:, 0:1],
        state[:, 1:2],
        state[:, 2:3],
        state[:, 3:4],
        state[:, 4:5],
    )
    face_specs = {
        "west": (4, 1),
        "east": (4, -1),
        "south": (3, 1),
        "north": (3, -1),
        "bottom": (2, 1),
        "top": (2, -1),
    }
    faces: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, (axis, shift) in face_specs.items():
        interpolated: list[torch.Tensor] = []
        boundary_mask: torch.Tensor | None = None
        for quantity in (pressure, velocity_x, velocity_y, velocity_z, scalar):
            neighbour, mask = _neighbour(quantity, axis, shift)
            boundary_mask = mask
            interpolated.append(0.5 * (quantity + neighbour))
        assert boundary_mask is not None
        pressure_face, velocity_x_face, velocity_y_face, velocity_z_face, scalar_face = interpolated
        prescribed_x = torch.zeros_like(velocity_x_face)
        prescribed_y = torch.zeros_like(velocity_y_face)
        prescribed_z = torch.zeros_like(velocity_z_face)
        prescribed_scalar = torch.full_like(scalar_face, case.wall_scalar)
        if name == "top":
            prescribed_x = torch.full_like(velocity_x_face, case.lid_velocity)
            prescribed_scalar = torch.full_like(scalar_face, case.top_scalar)
        faces[name] = (
            torch.where(boundary_mask, pressure, pressure_face),
            torch.where(boundary_mask, prescribed_x, velocity_x_face),
            torch.where(boundary_mask, prescribed_y, velocity_y_face),
            torch.where(boundary_mask, prescribed_z, velocity_z_face),
            torch.where(boundary_mask, prescribed_scalar, scalar_face),
            boundary_mask,
            axis,
            shift,
        )

    names = ("west", "east", "south", "north", "bottom", "top")
    unpacked = [faces[name] for name in names]
    pressure_faces = [item[0] for item in unpacked]
    x_faces = [item[1] for item in unpacked]
    y_faces = [item[2] for item in unpacked]
    z_faces = [item[3] for item in unpacked]
    scalar_faces = [item[4] for item in unpacked]
    masks = [item[5] for item in unpacked]
    axes = [int(item[6]) for item in unpacked]
    shifts = [int(item[7]) for item in unpacked]

    area = case.face_area
    fluxes = [
        -area * x_faces[0],
        area * x_faces[1],
        -area * y_faces[2],
        area * y_faces[3],
        -area * z_faces[4],
        area * z_faces[5],
    ]
    continuity = sum(fluxes)
    momentum_x = case.density * sum(
        flux * face for flux, face in zip(fluxes, x_faces, strict=True)
    ) + area * (pressure_faces[1] - pressure_faces[0])
    momentum_y = case.density * sum(
        flux * face for flux, face in zip(fluxes, y_faces, strict=True)
    ) + area * (pressure_faces[3] - pressure_faces[2])
    momentum_z = case.density * sum(
        flux * face for flux, face in zip(fluxes, z_faces, strict=True)
    ) + area * (pressure_faces[5] - pressure_faces[4])
    scalar_residual = sum(
        flux * face for flux, face in zip(fluxes, scalar_faces, strict=True)
    )

    def diffusion(quantity: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        return sum(
            _normal_gradient(quantity, face, mask, axis, shift, case.spacing)
            for face, mask, axis, shift in zip(values, masks, axes, shifts, strict=True)
        )

    momentum_x = momentum_x - case.viscosity * area * diffusion(velocity_x, x_faces)
    momentum_y = momentum_y - case.viscosity * area * diffusion(velocity_y, y_faces)
    momentum_z = momentum_z - case.viscosity * area * diffusion(velocity_z, z_faces)
    scalar_residual = scalar_residual - (1.0 / case.peclet) * area * diffusion(scalar, scalar_faces)

    if previous_state is not None:
        if time_step is None or time_step <= 0.0:
            raise ValueError("a positive time_step is required with previous_state")
        for channel, residual_name in ((1, "x"), (2, "y"), (3, "z")):
            derivative = case.density * case.cell_volume * (
                state[:, channel : channel + 1] - previous_state[:, channel : channel + 1]
            ) / time_step
            if residual_name == "x":
                momentum_x = momentum_x + derivative
            elif residual_name == "y":
                momentum_y = momentum_y + derivative
            else:
                momentum_z = momentum_z + derivative
        scalar_residual = scalar_residual + case.cell_volume * (
            scalar - previous_state[:, 4:5]
        ) / time_step

    return torch.cat(
        [continuity, momentum_x, momentum_y, momentum_z, scalar_residual], dim=1
    ) / case.cell_volume


def finite_volume_residual(
    state: torch.Tensor,
    case: CaseConfig,
    previous_state: torch.Tensor | None = None,
    time_step: float | None = None,
) -> torch.Tensor:
    """Assemble continuity, momentum and passive-scalar residuals.

    A supplied ``previous_state`` activates the backward-Euler accumulation
    terms. Reference solutions and target fields are never accepted here.
    """

    expected_shape = (1, case.state_channels, *case.spatial_shape)
    if tuple(state.shape) != expected_shape:
        raise ValueError(f"state shape must be {expected_shape}, got {tuple(state.shape)}")
    if previous_state is not None:
        if tuple(previous_state.shape) != expected_shape:
            raise ValueError("previous_state must have the same shape as state")
        previous_state = previous_state.to(device=state.device, dtype=state.dtype)
    if case.dimension == 2:
        return _residual_2d(state, case, previous_state, time_step)
    return _residual_3d(state, case, previous_state, time_step)


def finite_volume_objective(
    state: torch.Tensor,
    case: CaseConfig,
    previous_state: torch.Tensor | None = None,
    time_step: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the componentwise L1 finite-volume objective and diagnostics."""

    residual = finite_volume_residual(state, case, previous_state, time_step)
    reduce_dimensions = tuple(range(0, residual.ndim))
    component_values = residual.abs().mean(dim=tuple(index for index in reduce_dimensions if index != 1))
    pressure_gauge = state[:, 0:1].mean().square()
    objective = component_values.sum() + case.pressure_gauge_weight * pressure_gauge
    names = ["continuity", "momentum_x", "momentum_y"]
    if case.dimension == 3:
        names.append("momentum_z")
    names.append("passive_scalar")
    metrics = {
        name: float(value.detach().cpu())
        for name, value in zip(names, component_values, strict=True)
    }
    metrics["pressure_gauge"] = float(pressure_gauge.detach().cpu())
    metrics["objective"] = float(objective.detach().cpu())
    return objective, metrics
