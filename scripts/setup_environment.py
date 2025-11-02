"""_summary_
setup_environment.py

Initializes the CARLA simulation environment.

Connects to the local CARLA server, loads a specified map, stabilizes the world, and applies predefined weather parameters.

Usage:
        Run this script directly or via `client/scripts/launch_world.ps1` to set up the CARLA environment.
"""

import logging
import sys
import time

import carla


def main():
    """
    Initialize and configure the CARLA simulation environment based on the project specifications.

    Establishes a connection to the local CARLA server, loads the specified map,
    applies predefined weather conditions, and disables all traffic lights.

    Raises:
        Exception: If an error occurs during the environment setup.
    """
    HOST = "localhost"
    PORT = 2000
    WORLD_MAP = "Town01_Opt"
    TRAFFIC_LIGHT_STATE = carla.TrafficLightState.Off
    TRAFFIC_LIGHT_FREEZE = True

    try:
        # Connect to the CARLA server
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)

        # Load the environment map
        # world = client.get_world()
        world = client.load_world(
            map_name=WORLD_MAP, reset_settings=True, map_layers=carla.MapLayer.All
        )
        logging.info(f"{WORLD_MAP} loaded.")

        # Wait briefly to ensure the world is fully initialized
        for _ in range(1):
            world.wait_for_tick()

        # Configure weather parameters for environmental realism
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=14.0,
            sun_azimuth_angle=180.0,
            sun_altitude_angle=15.0,
            fog_density=3.0,
            fog_distance=40.0,
            wetness=0.0,
            fog_falloff=0.5,
            scattering_intensity=0.3,
            mie_scattering_scale=0.05,
            rayleigh_scattering_scale=0.0331,
        )

        # Update the environment weather
        world.set_weather(weather)
        logging.info("Weather updated.")

        # Wait briefly to ensure the world is fully initialized
        for _ in range(1):
            world.wait_for_tick()

        # Retrieve all traffic light actors from the simulation world
        traffic_lights = world.get_actors().filter("traffic.traffic_light")

        # Disable and freeze all traffic lights
        for light in traffic_lights:
            light.set_state(TRAFFIC_LIGHT_STATE)
            light.freeze(TRAFFIC_LIGHT_FREEZE)

        logging.info("Traffic lights disabled.")

        # Wait briefly to ensure the world is fully initialized
        for _ in range(1):
            world.wait_for_tick()

    except Exception as _:
        logging.critical("An unhandled exception occurred!", exc_info=True)
    finally:
        time.sleep(0.5)
        logging.info("Setup complete.\n")


# Entry point: configure logging and run environment setup
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
