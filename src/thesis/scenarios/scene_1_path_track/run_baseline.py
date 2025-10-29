import logging
import os
import sys
import time

# NOTE: Update the path below to your CARLA 0.9.15 installation folder containing 'PythonAPI\carla'.
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)

import carla
from agents.navigation.basic_agent import BasicAgent

# Add project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.helpers import actor_cleanup, draw_spawn_points  # noqa: E402, F401


def main():
    SYNCHRONOUS = False
    HOST = "localhost"
    PORT = 2000
    EGO_ACTIVE = False
    DELTA_SECONDS = 0.05
    SERVER_TIMEOUT = 10.0
    DISPLAY_TELEMETRY = False
    VEHICLE_LIST = []

    # Set up ego vehicle lights
    EGO_LIGHT_MASK = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

    # Indices specific to Town01_Opt only
    EGO_START_INDEX = 182
    EGO_STOP_INDEX = 77

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
        # draw_spawn_points(world, spawn_points)

        # Set up ego vehicle navigation endpoints
        EGO_START_POINT = spawn_points[EGO_START_INDEX]
        EGO_STOP_POINT = spawn_points[EGO_STOP_INDEX]
        EGO_ORIGIN = EGO_START_POINT.location
        EGO_DESTINATION = EGO_STOP_POINT.location

        # Define ego vehicle
        EGO_BP = blueprint_library.find("vehicle.mercedes.coupe_2020")
        if EGO_BP.has_attribute("role_name"):
            EGO_BP.set_attribute("role_name", "hero")
        if EGO_BP.has_attribute("color"):
            EGO_BP.set_attribute("color", "0,21,81")

        # Spawn ego vehicle
        EGO_VEHICLE = world.spawn_actor(EGO_BP, EGO_START_POINT)
        EGO_VEHICLE.set_light_state(carla.VehicleLightState(EGO_LIGHT_MASK))
        EGO_VEHICLE.show_debug_telemetry(DISPLAY_TELEMETRY)
        EGO_ACTIVE = True
        VEHICLE_LIST.append(EGO_VEHICLE)
        logging.info(f"Spawned Ego {EGO_VEHICLE.type_id} at Origin {EGO_ORIGIN}.")

        # Wait briefly to ensure the world is fully initialized
        for _ in range(1):
            world.tick()

        # Set up ego basic agent
        EGO_AGENT = BasicAgent(
            vehicle=EGO_VEHICLE,
            target_speed=50,
            opt_dict={
                "ignore_traffic_lights": True,
                "ignore_stop_signs": True,
                "ignore_vehicles": False,
                "follow_speed_limits": False,
            },
        )
        logging.debug("Ego Basic Agent initialized.")

        # Set up ego agent navigation plan
        EGO_AGENT.set_destination(EGO_DESTINATION)
        logging.info(f"Ego Basic Agent set to destination {EGO_DESTINATION}.")

        # Wait briefly to ensure the world is fully initialized
        for _ in range(1):
            world.tick()

        # Run game loop
        while True:
            world.tick()
            if EGO_ACTIVE:
                # Set ego agent's target speed based on the maximum predefined OpenDRIVE lane speed limits
                EGO_AGENT.set_target_speed(EGO_VEHICLE.get_speed_limit())

                # Set up spectator camera
                EGO_TRANSFORM = EGO_VEHICLE.get_transform()
                # For chase camera view
                spectator_transform = carla.Transform(
                    EGO_TRANSFORM.transform(carla.Location(x=-4, z=2)),
                    EGO_TRANSFORM.rotation,
                )
                spectator.set_transform(spectator_transform)

                # Compute and baseline vehicle controls
                ego_control = EGO_AGENT.run_step()
                EGO_VEHICLE.apply_control(ego_control)

            # Terminate the game loop
            if EGO_ACTIVE and EGO_AGENT.done():
                logging.info("Ego Basic Agent reached destination.")
                time.sleep(0.017)
                VEHICLE_LIST.remove(EGO_VEHICLE)
                EGO_ACTIVE = False
                break

            time.sleep(0.017)

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
