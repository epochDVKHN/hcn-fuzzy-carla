"""_summary_
visualize_spawn_points.py

Visualize default CARLA spawn point locations.

This script connects to a running CARLA simulator, retrieves the current map's spawn points, and displays them as labeled markers. It also positions the spectator in a top-down view for spatial reference.

Usage:
    Run directly or via `client/scripts/launch_spawn_visualization.ps1` to visualize spawn points in the active CARLA map.

"""

import logging
import sys
import time
from pathlib import Path

import carla

# Add project root path for module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[1] / "src" / "thesis"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import draw_spawn_waypoints, world_snapshot_tick  # noqa: E402


def main():
    """
    Initialize a CARLA simulation and visualize spawn points.

    Connects to a CARLA server, retrieves the current map and its spawn points, draws them as labeled markers, and positions the spectator above a reference spawn point for top-down visualization.

    Raises:
        Exception: If any error occurs during setup or connection.
    """
    HOST = "localhost"
    PORT = 2000
    DISPLAY_TIME = 300  # Duration (in seconds)
    REFERENCE_INDEX = 0

    try:
        # Connect to the CARLA server
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)

        # Retrieve world and map data
        world = client.get_world()
        map = world.get_map()
        spectator = world.get_spectator()
        spawn_points = map.get_spawn_points()

        # Get the CARLA Town Map Name
        TOWN_MAP_NAME = map.name.rsplit("/", 1)[-1]

        # Draw numeric labels at all spawn point locations
        draw_spawn_waypoints(world, spawn_points, DISPLAY_TIME)

        logging.info(f"{TOWN_MAP_NAME} spawn points loaded.")

        # Set spectator to top-down view of a reference spawn point
        reference_point = spawn_points[REFERENCE_INDEX]
        SPECTATOR_TRANSFORM = carla.Transform(
            reference_point.transform(carla.Location(x=-4, y=0, z=50)),
            carla.Rotation(pitch=-90, yaw=reference_point.rotation.yaw, roll=0),
        )
        spectator.set_transform(SPECTATOR_TRANSFORM)

        # Wait briefly to ensure the world state stabilizes
        world_snapshot_tick(world)

    except Exception as _:
        logging.critical(
            "Simulation failed due to an unhandled exception!", exc_info=True
        )
    finally:
        time.sleep(0.5)
        logging.info("Spawn points visualization completed.\n")


# Entry point: configure logging and run spawn points visualization
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} | {levelname:<8} | {name} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    main()
