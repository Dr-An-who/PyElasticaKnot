"""Example: build a rod from a list of 3D points and impose a displacement boundary condition.

The script demonstrates how to
* read a text file containing the coordinates of a space curve,
* initialize a Cosserat rod whose reference configuration follows the curve,
* prescribe a displacement on the rod tip via a time dependent boundary condition, and
* report the resulting displacement and reaction forces at both ends of the rod.

Run the example from the repository root with::

    uv run python examples/BoundaryConditionsCases/rod_from_points_displacement.py

The command line interface provides options to select a different point file,
the imposed displacement, and the integration parameters. Use ``--help`` to
see all available flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import elastica as ea


class CurveDisplacementSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    """Minimal simulator used in this example."""


@dataclass
class SimulationResults:
    """Container for the quantities reported at the end of the run."""

    final_tip_displacement: np.ndarray
    left_reaction: np.ndarray
    right_reaction: np.ndarray


def read_points(path: Path) -> np.ndarray:
    """Read xyz coordinates from ``path``.

    Each line of the text file must contain three floating point numbers
    separated by whitespace. The resulting array has shape ``(n_points, 3)``.
    """

    data = []
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Expected exactly three coordinates per line, found: '{stripped}'"
                )
            data.append(tuple(float(value) for value in parts))

    coordinates = np.asarray(data, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "The input file must contain at least two rows with three coordinates each."
        )
    if coordinates.shape[0] < 2:
        raise ValueError("At least two points are required to define a centerline.")

    return coordinates


def _compute_initial_direction(points: np.ndarray) -> np.ndarray:
    """Return a unit vector aligned with the first segment of the curve."""

    direction = points[1] - points[0]
    norm = np.linalg.norm(direction)
    if norm <= np.finfo(float).eps:
        raise ValueError("The first two points in the curve must be distinct.")
    return direction / norm


def _compute_reference_normal(direction: np.ndarray) -> np.ndarray:
    """Construct a normal vector that is not colinear with ``direction``."""

    trial_vectors: Iterable[np.ndarray] = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    for trial in trial_vectors:
        candidate = np.cross(direction, trial)
        norm = np.linalg.norm(candidate)
        if norm > np.finfo(float).eps:
            return candidate / norm
    raise ValueError("Unable to construct a normal vector for the provided direction.")


class TipDisplacementBC(ea.boundary_conditions.ConstraintBase):
    """Move the constrained node along a straight line in time."""

    def __init__(
        self,
        initial_position: np.ndarray,
        imposed_displacement: np.ndarray,
        ramp_up_time: float,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._initial_position = initial_position.reshape(3)
        self._imposed_displacement = imposed_displacement.reshape(3)
        self._ramp_up_time = float(ramp_up_time)
        if self._ramp_up_time <= 0.0:
            self._ramp_up_time = 0.0
            self._velocity = np.zeros(3)
        else:
            self._velocity = self._imposed_displacement / self._ramp_up_time

    def _compute_target(self, time: float) -> np.ndarray:
        if self._ramp_up_time == 0.0:
            alpha = 1.0
        else:
            alpha = min(max(time / self._ramp_up_time, 0.0), 1.0)
        return self._initial_position + alpha * self._imposed_displacement

    def constrain_values(
        self, system: ea.typing.SystemType, time: np.float64
    ) -> None:  # noqa: D401 - see base class
        target = self._compute_target(float(time))
        for index in self.constrained_position_idx:
            system.position_collection[:, index] = target

    def constrain_rates(
        self, system: ea.typing.SystemType, time: np.float64
    ) -> None:  # noqa: D401 - see base class
        if self._ramp_up_time == 0.0:
            velocity = np.zeros(3)
        else:
            velocity = self._velocity if time < self._ramp_up_time else np.zeros(3)
        for index in self.constrained_position_idx:
            system.velocity_collection[:, index] = velocity


class ReactionCallBack(ea.CallBackBaseClass):
    """Store displacement and reaction forces during the simulation."""

    def __init__(
        self,
        step_skip: int,
        callback_params: dict[str, list[np.ndarray]],
        initial_start: np.ndarray,
        initial_end: np.ndarray,
    ) -> None:
        super().__init__()
        self.every = max(step_skip, 1)
        self.callback_params = callback_params
        self._initial_start = initial_start.copy()
        self._initial_end = initial_end.copy()

    def make_callback(
        self, system: ea.typing.RodType, time: np.float64, current_step: int
    ) -> None:  # noqa: D401 - see base class
        if current_step % self.every != 0:
            return
        self.callback_params.setdefault("time", []).append(float(time))
        displacement = system.position_collection[:, -1] - self._initial_end
        self.callback_params.setdefault("tip_displacement", []).append(displacement.copy())
        self.callback_params.setdefault("left_reaction", []).append(
            system.internal_forces[:, 0].copy()
        )
        self.callback_params.setdefault("right_reaction", []).append(
            system.internal_forces[:, -1].copy()
        )


def build_rod_from_points(
    points: np.ndarray,
    density: float,
    youngs_modulus: float,
    radius: float,
) -> tuple[ea.typing.RodType, float]:
    """Instantiate a Cosserat rod that follows ``points``.

    Returns the rod and the total arc length of the input curve.
    """

    n_elems = points.shape[0] - 1
    direction = _compute_initial_direction(points)
    normal = _compute_reference_normal(direction)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    base_length = float(np.sum(segment_lengths))
    shear_modulus = youngs_modulus / 1.5  # corresponds to Poisson's ratio of 0.5

    rod = ea.CosseratRod.straight_rod(
        n_elems,
        start=points[0],
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
        position=points.T,
    )
    return rod, base_length


def run_simulation(
    points: np.ndarray,
    imposed_displacement: np.ndarray,
    density: float,
    youngs_modulus: float,
    radius: float,
    final_time: float,
    ramp_time: float,
    time_step: float,
    damping_constant: float,
    step_skip: int,
) -> SimulationResults:
    """Integrate the elastica system and report the final state."""

    simulator = CurveDisplacementSimulator()
    rod, base_length = build_rod_from_points(points, density, youngs_modulus, radius)
    simulator.append(rod)

    simulator.constrain(rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        fixed_position=points[0],
        fixed_directors=rod.director_collection[:, :, 0],
    )
    simulator.constrain(rod).using(
        TipDisplacementBC,
        constrained_position_idx=(-1,),
        initial_position=points[-1],
        imposed_displacement=imposed_displacement,
        ramp_up_time=ramp_time,
    )

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    history: dict[str, list[np.ndarray]] = {}
    simulator.collect_diagnostics(rod).using(
        ReactionCallBack,
        step_skip=step_skip,
        callback_params=history,
        initial_start=points[0],
        initial_end=points[-1],
    )

    simulator.finalize()

    timestepper = ea.PositionVerlet()
    total_steps = int(final_time / time_step)
    ea.integrate(timestepper, simulator, final_time, total_steps)

    final_disp = history["tip_displacement"][-1]
    left_reaction = history["left_reaction"][-1]
    right_reaction = history["right_reaction"][-1]
    return SimulationResults(final_disp, left_reaction, right_reaction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "space_curve_points.pts",
        help="Path to the text file containing the curve coordinates.",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        nargs=3,
        default=(0.0, -0.02, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Imposed displacement applied to the rod tip (in meters).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.003,
        help="Uniform radius of the rod (in meters).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1200.0,
        help="Material density (in kg/m^3).",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=5.0e5,
        help="Young's modulus (in Pascals).",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=1.0,
        help="Total simulated time (in seconds).",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.5,
        help="Time required to reach the prescribed displacement (in seconds).",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=5.0e-4,
        help="Time step used by the integrator (in seconds).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=2.0,
        help="Damping constant used by the analytical linear damper.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Store diagnostics every N integration steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = read_points(args.points)
    imposed_displacement = np.asarray(args.displacement, dtype=np.float64)
    results = run_simulation(
        points=points,
        imposed_displacement=imposed_displacement,
        density=args.density,
        youngs_modulus=args.youngs_modulus,
        radius=args.radius,
        final_time=args.final_time,
        ramp_time=args.ramp_time,
        time_step=args.time_step,
        damping_constant=args.damping,
        step_skip=args.log_every,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("Final tip displacement [m]:", results.final_tip_displacement)
    print("Reaction at the clamped end [N]:", results.left_reaction)
    print("Reaction at the displaced end [N]:", results.right_reaction)


if __name__ == "__main__":
    main()
