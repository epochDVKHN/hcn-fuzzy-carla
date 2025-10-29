import logging
import sys
import time

import carla
from utils.helpers import actor_cleanup, draw_spawn_points


def main():
    SYNCHRONOUS = False
    HOST = "localhost"
    PORT = 2000
    EGO_ACTIVE = False
    DELTA_SECONDS = 0.05  # 20 FPS
    SERVER_TIMEOUT = 10.0

    # Indices specific to Town01_Opt only
    EGO_START_INDEX = 184
    EGO_STOP_INDEX = 47

    try:
        # Connect to CARLA server
        client = carla.Client(HOST, PORT)
        client.set_timeout(SERVER_TIMEOUT)

        # Preliminary setup variables
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        original_settings = world.get_settings()
        spectator = world.get_spectator()
        spawn_points = map.get_spawn_points()

        logging.info("Simulation started.")

        # Enable synchronous mode for determinism
        current_settings = world.get_settings()
        current_settings.synchronous_mode = True
        current_settings.fixed_delta_seconds = DELTA_SECONDS
        world.apply_settings(current_settings)
        SYNCHRONOUS = True
        logging.debug("Synchronous mode enabled.")

        # Draw the default spawn point locations as numbers in the map
        draw_spawn_points(world, spawn_points)

    except KeyboardInterrupt:
        logging.warning("Simulation interrupted by user!")
    except Exception as _:
        logging.critical("An unhandled exception occurred!", exc_info=True)
    finally:
        # Disable synchronous mode and restore original settings
        if SYNCHRONOUS:
            world.apply_settings(original_settings)
            SYNCHRONOUS = False
            logging.debug("Synchronous mode disabled.")
        # Stop and destroy all actors in the world
        actor_cleanup(world)
        time.sleep(0.5)

        logging.info("Simulation stopped.\n")


# Entry point: configure logging and run baseline path tracking scenario
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
