import logging
import sys
import time
from collections import deque
from pathlib import Path

# NOTE: Update the path below to your CARLA 0.9.15 installation folder containing 'PythonAPI\carla'.
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)

import carla
from agents.navigation.basic_agent import BasicAgent

# Add project root path for module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import (  # noqa: E402
    SpectatorCameraPositions,
    actor_cleanup,
    draw_plan_waypoints,
    generate_custom_route,
    get_route_length,
    set_spectator_camera,
    set_vehicle_lights,
)


def main():
    SYNCHRONOUS = False
    HOST = "localhost"
    PORT = 2000
    EGO_ACTIVE = False
    DELTA_SECONDS = 0.05
    SERVER_TIMEOUT = 10.0
    DISPLAY_TELEMETRY = False
    DRAW_PLAN_WAYPOINTS = False
    CUSTOM_ROUTE_NAVIGATION = False
    VEHICLE_LIST = []

    # Store recent waypoints for local planner history
    ego_step_history = deque(maxlen=12)

    # Set up default ego vehicle lights
    EGO_DEFAULT_LIGHT_MASK = (
        carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
    )
    # NOTE: Custom route indices for the CARLA Town01_Opt map
    # Format: [Start Point, Spawn Point 2, ..., End Point]
    # Intermediate points define route segments; only key points (e.g., after turns) are needed.
    # Use `launch_spawn_visualization.ps1` to view spawn points and choose an appropriate start index.
    EGO_ROUTE_INDICES = [118, 141, 78, 173, 227, 15, 0, 140, 16, 129, 96, 119, 88]
    # EGO_ROUTE_INDICES = [184, 182, 47, 36]

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

        # Set up ego vehicle navigation endpoints
        EGO_START_POINT = spawn_points[EGO_ROUTE_INDICES[0]]
        EGO_STOP_POINT = spawn_points[EGO_ROUTE_INDICES[-1]]
        EGO_ORIGIN = EGO_START_POINT.location
        EGO_DESTINATION = EGO_STOP_POINT.location

        # Define ego vehicle
        EGO_BP = blueprint_library.find("vehicle.mercedes.coupe_2020")
        if EGO_BP.has_attribute("role_name"):
            EGO_BP.set_attribute("role_name", "hero")
        if EGO_BP.has_attribute("color"):
            EGO_BP.set_attribute("color", "0,21,81")

        # Spawn ego vehicle and add vehicle functionalities
        EGO_VEHICLE = world.spawn_actor(EGO_BP, EGO_START_POINT)
        EGO_VEHICLE.set_light_state(carla.VehicleLightState(EGO_DEFAULT_LIGHT_MASK))
        EGO_VEHICLE.show_debug_telemetry(DISPLAY_TELEMETRY)

        EGO_ACTIVE = True
        VEHICLE_LIST.append(EGO_VEHICLE)
        logging.info(f"Ego vehicle ({EGO_VEHICLE.type_id}) spawned at {EGO_ORIGIN}.")

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

        if not CUSTOM_ROUTE_NAVIGATION:
            # Use A* global route planner for shortest-path navigation
            EGO_AGENT.set_destination(EGO_DESTINATION)
        else:
            # Use predefined custom route for navigation
            EGO_ROUTE_PLAN = generate_custom_route(EGO_ROUTE_INDICES, spawn_points, map)
            EGO_AGENT.set_global_plan(EGO_ROUTE_PLAN)

        ROUTE_TYPE = "custom" if CUSTOM_ROUTE_NAVIGATION else "A*"
        logging.info(
            f"Ego vehicle navigation initialized ({ROUTE_TYPE} route) toward {EGO_DESTINATION}."
        )
        logging.info(f"Total route length: {get_route_length(EGO_AGENT):.2f} m.")

        # Wait briefly to ensure the world is fully initialized
        for _ in range(1):
            world.tick()

        # Run game loop
        while True:
            server_frame_ID = world.tick()

            if EGO_ACTIVE:
                # Set ego agent's target speed based on the maximum predefined OpenDRIVE lane speed limits
                EGO_AGENT.set_target_speed(EGO_VEHICLE.get_speed_limit())

                # Set up spectator camera
                spectator_transform = set_spectator_camera(
                    EGO_VEHICLE,
                    SpectatorCameraPositions.LOW_ANGLE_CHASE_VIEW,
                )
                spectator.set_transform(spectator_transform)

                # Draw the plan path in the map
                if DRAW_PLAN_WAYPOINTS:
                    draw_plan_waypoints(world, EGO_AGENT, server_frame_ID)

                # Compute baseline vehicle controls
                ego_control = EGO_AGENT.run_step()

                # Set up ego vehicle lights
                ego_new_light_mask = set_vehicle_lights(
                    EGO_DEFAULT_LIGHT_MASK,
                    ego_control,
                    EGO_AGENT,
                    ego_step_history,
                )
                EGO_VEHICLE.set_light_state(ego_new_light_mask)

                # Apply baseline vehicle controls
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
        logging.critical(
            "Simulation failed due to an unhandled exception.", exc_info=True
        )
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
