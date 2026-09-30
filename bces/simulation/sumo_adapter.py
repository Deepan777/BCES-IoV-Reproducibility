"""Pinned SUMO/TraCI loading and deterministic compact state extraction."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import sumo  # type: ignore[import-untyped]
from bces.oracle.world import EgoState, WorldObject


def load_traci():
    tools = str(Path(sumo.SUMO_HOME) / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import traci  # type: ignore[import-not-found]

    return traci


def sumo_binary() -> Path:
    path = Path(sumo.SUMO_HOME) / "bin" / "sumo.exe"
    if not path.is_file():
        raise FileNotFoundError("pinned headless SUMO binary is unavailable")
    return path


def sumo_version() -> str:
    import subprocess

    result = subprocess.run(
        [str(sumo_binary()), "--version"], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()[0].split()[-1]


def start_connection(
    *, network: Path, label: str, seed: int, step_s: float, state: Path | None = None
):
    traci = load_traci()
    command = [
        str(sumo_binary()), "-n", str(network), "--step-length", str(step_s),
        "--seed", str(seed), "--no-step-log", "true", "--no-warnings", "true",
        "--time-to-teleport", "-1", "--collision.action", "warn",
    ]
    if state is not None:
        command.extend(("--load-state", str(state)))
    traci.start(command, label=label)
    return traci.getConnection(label)


def heading_rad(angle_degrees: float) -> float:
    return math.radians(90.0 - angle_degrees)


def ego_state(connection: Any, ego_id: str = "ego") -> EgoState:
    x_m, y_m = connection.vehicle.getPosition(ego_id)
    return EgoState(
        x_m=x_m,
        y_m=y_m,
        speed_mps=connection.vehicle.getSpeed(ego_id),
        heading_rad=heading_rad(connection.vehicle.getAngle(ego_id)),
        acceleration_mps2=connection.vehicle.getAcceleration(ego_id),
        curvature_inv_m=0.0,
        length_m=connection.vehicle.getLength(ego_id),
        width_m=connection.vehicle.getWidth(ego_id),
    )


def world_objects(connection: Any, *, exclude: str = "ego") -> tuple[WorldObject, ...]:
    rows = []
    for vehicle_id in sorted(connection.vehicle.getIDList()):
        if vehicle_id == exclude:
            continue
        x_m, y_m = connection.vehicle.getPosition(vehicle_id)
        rows.append(
            WorldObject(
                track_id=vehicle_id,
                x_m=x_m,
                y_m=y_m,
                vx_mps=connection.vehicle.getSpeed(vehicle_id) * math.cos(heading_rad(connection.vehicle.getAngle(vehicle_id))),
                vy_mps=connection.vehicle.getSpeed(vehicle_id) * math.sin(heading_rad(connection.vehicle.getAngle(vehicle_id))),
                heading_rad=heading_rad(connection.vehicle.getAngle(vehicle_id)),
                length_m=connection.vehicle.getLength(vehicle_id),
                width_m=connection.vehicle.getWidth(vehicle_id),
                source="sumo",
            )
        )
    for person_id in sorted(connection.person.getIDList()):
        x_m, y_m = connection.person.getPosition(person_id)
        angle = heading_rad(connection.person.getAngle(person_id))
        speed = connection.person.getSpeed(person_id)
        rows.append(
            WorldObject(
                track_id=person_id,
                x_m=x_m,
                y_m=y_m,
                vx_mps=speed * math.cos(angle),
                vy_mps=speed * math.sin(angle),
                heading_rad=angle,
                length_m=0.6,
                width_m=0.6,
                class_id=1,
                source="sumo",
            )
        )
    return tuple(rows)


def compact_state(connection: Any, *, exclude_ego: bool = True) -> dict[str, Any]:
    vehicles = {}
    for vehicle_id in sorted(connection.vehicle.getIDList()):
        if exclude_ego and vehicle_id == "ego":
            continue
        vehicles[vehicle_id] = {
            "position": tuple(round(v, 6) for v in connection.vehicle.getPosition(vehicle_id)),
            "speed": round(connection.vehicle.getSpeed(vehicle_id), 6),
            "lane": connection.vehicle.getLaneID(vehicle_id),
            "route_index": connection.vehicle.getRouteIndex(vehicle_id),
        }
    persons = {
        person_id: {
            "position": tuple(round(v, 6) for v in connection.person.getPosition(person_id)),
            "speed": round(connection.person.getSpeed(person_id), 6),
            "road": connection.person.getRoadID(person_id),
        }
        for person_id in sorted(connection.person.getIDList())
    }
    signals = {
        signal_id: connection.trafficlight.getRedYellowGreenState(signal_id)
        for signal_id in sorted(connection.trafficlight.getIDList())
    }
    return {
        "time_s": round(connection.simulation.getTime(), 6),
        "vehicles": vehicles,
        "persons": persons,
        "signals": signals,
    }


def state_hash(connection: Any) -> str:
    payload = json.dumps(compact_state(connection), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
