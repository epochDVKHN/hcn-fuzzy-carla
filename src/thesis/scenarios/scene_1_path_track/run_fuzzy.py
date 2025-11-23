import logging
import sys
import time
from collections import deque
from pathlib import Path

import carla

# Include the project root for relative module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from custom_agents.fuzzy_agent import FuzzyControlAgent  # noqa: E402
from utils.helpers import (  # noqa: E402
    SpectatorCameraPositions,
    compute_client_fps,
    compute_server_fps,
    draw_plan_waypoints,
    generate_custom_route,
    get_actor_display_name,
    get_route_length,
    halt_vehicle,
    set_spectator_camera,
    set_vehicle_lights,
    update_vehicle_physics,
    world_actor_cleanup,
    world_snapshot_tick,
)
from utils.sensors import (  # noqa: E402
    CollisionSensor,
    GNSSSensor,
    IMUSensor,
    LaneInvasionSensor,
)


def main():
    # ------------- Simulation and Connection Settings -------------
    SYNCHRONOUS = False
    HOST = "localhost"
    PORT = 2000
    SERVER_DELTA_SECONDS = 1.0 / 20  # Fixed delta for synchronous mode
    SERVER_TIMEOUT = 10.0
    CLIENT_DELTA_SLEEP = 0.02  # Client-side sleep to improve visualization smoothness, recommended range [0.02–0.04] seconds
    DISPLAY_TELEMETRY = False  # Show telemetry info on vehicle
    DRAW_PLAN_WAYPOINTS = False  # Visualize local route plan
    CUSTOM_ROUTE_NAVIGATION = False  # Use predefined route if True
    VEHICLE_LIST = []  # Track spawned vehicles
    SENSOR_LIST = []  # Track spawned sensors
    EGO_TIMEOUT = 2.5  # Pause duration at simulation start
    LOG_FPS_FREQ = 300  # Frequency of FPS logging

    # Store recent ego waypoints history for light logic
    ego_step_history = deque(maxlen=12)

    # Default lights for ego vehicle
    EGO_DEFAULT_LIGHT_MASK = (
        carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
    )

    # Ego vehicle sensors
    EGO_COLLISION_SENSOR = None
    EGO_GNSS_SENSOR = None
    EGO_IMU_SENSOR = None
    EGO_LANE_INVASION_SENSOR = None

    # Custom spawn point indices for defining a route in Town01_Opt
    # EGO_ROUTE_INDICES = [118, 141, 78, 173, 227, 15, 0, 140, 16, 129, 96, 119, 88]
    EGO_ROUTE_INDICES = [184, 182, 47, 36]

    try:
        # ----------------- Connect to CARLA Server -----------------
        client = carla.Client(HOST, PORT)
        client.set_timeout(SERVER_TIMEOUT)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        original_settings = world.get_settings()
        spectator = world.get_spectator()
        spawn_points = map.get_spawn_points()
        LOGGER.info("Simulation started.")

        # ----------------- Enable synchronous mode -----------------
        current_settings = world.get_settings()
        current_settings.synchronous_mode = True
        current_settings.fixed_delta_seconds = SERVER_DELTA_SECONDS
        world.apply_settings(current_settings)
        SYNCHRONOUS = True
        LOGGER.debug("Synchronous mode enabled.")
        server_fps = compute_server_fps(SERVER_DELTA_SECONDS)

        # ----------------- Ego vehicle setup -----------------
        EGO_START_POINT = spawn_points[EGO_ROUTE_INDICES[0]]
        EGO_STOP_POINT = spawn_points[EGO_ROUTE_INDICES[-1]]
        EGO_ORIGIN = EGO_START_POINT.location
        EGO_DESTINATION = EGO_STOP_POINT.location

        # Select vehicle blueprint and set attributes
        EGO_BP = blueprint_library.find("vehicle.mercedes.coupe_2020")
        if EGO_BP.has_attribute("role_name"):
            EGO_BP.set_attribute("role_name", "hero")
        if EGO_BP.has_attribute("color"):
            EGO_BP.set_attribute("color", "73,0,0")

        # Spawn ego vehicle
        EGO_VEHICLE = world.spawn_actor(EGO_BP, EGO_START_POINT)
        EGO_VEHICLE.set_light_state(carla.VehicleLightState(EGO_DEFAULT_LIGHT_MASK))
        EGO_VEHICLE.show_debug_telemetry(DISPLAY_TELEMETRY)
        VEHICLE_LIST.append(EGO_VEHICLE)
        LOGGER.info(
            f"Ego vehicle [{get_actor_display_name(EGO_VEHICLE)}] spawned at {EGO_ORIGIN}."
        )
        # Wait briefly to ensure the world state stabilizes
        world_snapshot_tick(world)

        # ----------------- Ego vehicle sensors -----------------
        EGO_COLLISION_SENSOR = CollisionSensor(EGO_VEHICLE)
        EGO_GNSS_SENSOR = GNSSSensor(EGO_VEHICLE)
        EGO_IMU_SENSOR = IMUSensor(EGO_VEHICLE)
        EGO_LANE_INVASION_SENSOR = LaneInvasionSensor(EGO_VEHICLE)
        SENSOR_LIST.extend(
            [
                EGO_COLLISION_SENSOR.sensor,
                EGO_GNSS_SENSOR.sensor,
                EGO_IMU_SENSOR.sensor,
                EGO_LANE_INVASION_SENSOR.sensor,
            ]
        )
        world_snapshot_tick(world)

        # Update vehicle physics before starting the agent
        update_vehicle_physics(EGO_VEHICLE)
        world_snapshot_tick(world)

        # ----------------- Initialize ego agent -----------------
        EGO_AGENT = FuzzyControlAgent(
            vehicle=EGO_VEHICLE,
            target_speed=50,
            opt_dict={
                "max_throttle": 1.0,  # Custom value
                "max_brake": 1.0,  # Custom value
                "max_steering": 0.8,
                "ignore_traffic_lights": True,
                "ignore_stop_signs": True,
                "ignore_vehicles": False,
                "follow_speed_limits": False,
                "offset": 0.0,
                "lateral_control_dict": {
                    "K_P": 1.95,
                    "K_I": 0.05,
                    "K_D": 0.2,
                    "dt": SERVER_DELTA_SECONDS,
                },
            },
        )
        LOGGER.debug("Ego [Fuzzy Agent] initialized.")
        world_snapshot_tick(world)

        # Set navigation plan: A* route or custom route
        if not CUSTOM_ROUTE_NAVIGATION:
            EGO_AGENT.set_destination(EGO_DESTINATION)
        else:
            EGO_ROUTE_PLAN = generate_custom_route(EGO_ROUTE_INDICES, spawn_points, map)
            EGO_AGENT.set_global_plan(EGO_ROUTE_PLAN)

        ROUTE_TYPE = "custom" if CUSTOM_ROUTE_NAVIGATION else "A*"
        LOGGER.info(
            f"Ego vehicle navigation initialized [{ROUTE_TYPE} route] toward {EGO_DESTINATION}."
        )
        LOGGER.info(f"Total route length: {get_route_length(EGO_AGENT):.2f} m.")
        world_snapshot_tick(world)

        # ----------------- Main simulation loop -----------------
        EGO_START_TIME = time.time()
        last_time = EGO_START_TIME
        end_of_route = False
        finish_start_time = None

        # Store previous steering to maintain lane during stop
        last_steer = EGO_VEHICLE.get_control().steer

        while True:
            # Advance the simulation and get the latest world snapshot
            world_snapshot = world_snapshot_tick(world)
            snapshot_frame_ID = world_snapshot.frame

            # Check if ego vehicle reached its destination
            if not end_of_route and EGO_AGENT.done():
                LOGGER.info("Ego [Basic Agent] reached destination.")
                end_of_route = True
                finish_start_time = time.time()

            # Stop simulation after timeout post reaching destination
            if end_of_route and time.time() - finish_start_time >= EGO_TIMEOUT:
                time.sleep(CLIENT_DELTA_SLEEP)
                break

            # Update spectator camera to follow ego vehicle
            spectator_transform = set_spectator_camera(
                EGO_VEHICLE,
                SpectatorCameraPositions.LOW_ANGLE_CHASE_VIEW,
            )
            spectator.set_transform(spectator_transform)

            # Determine ego control based on route status
            if end_of_route or time.time() - EGO_START_TIME < EGO_TIMEOUT:
                # Apply brake and maintain last steering
                ego_control = halt_vehicle(last_steer)
            else:
                # Set agent's target speed according to OpenDRIVE lane speed limits
                EGO_AGENT.set_target_speed(EGO_VEHICLE.get_speed_limit())

                # Optionally visualize planned route
                if DRAW_PLAN_WAYPOINTS:
                    draw_plan_waypoints(world, EGO_AGENT, snapshot_frame_ID)

                # Compute control commands for this step
                ego_control = EGO_AGENT.run_step(snapshot_frame_ID)

                # Compute lateral and heading deviations for monitoring
                # lateral_error, angular_error = compute_steering_deviations(EGO_AGENT)
                # if snapshot_frame_ID % 10 == 0:
                #     print(
                #         f"Lateral error: {lateral_error:.2f} | Angular error: {angular_error:.2f} | Steer: {ego_control.steer:.2f}"
                #     )

                # Update last applied steering
                last_steer = ego_control.steer

            # Update vehicle light states based on vehicle control and step history
            ego_new_light_mask = set_vehicle_lights(
                EGO_DEFAULT_LIGHT_MASK,
                ego_control,
                EGO_AGENT,
                ego_step_history,
            )
            EGO_VEHICLE.set_light_state(ego_new_light_mask)

            # Apply computed control to ego vehicle
            EGO_VEHICLE.apply_control(ego_control)

            # Sleep to regulate visualization update rate
            time.sleep(CLIENT_DELTA_SLEEP)

            # Compute client FPS for diagnostics
            current_time = time.time()
            client_fps = compute_client_fps(current_time, last_time)
            last_time = current_time
            if snapshot_frame_ID % LOG_FPS_FREQ == 0:
                LOGGER.debug(
                    f"FPS [Server = {server_fps:.1f}, Client = {client_fps:.1f}]."
                )

    except KeyboardInterrupt:
        LOGGER.warning("Simulation interrupted by user!")
    except Exception as _:
        LOGGER.critical(
            "Simulation failed due to an unhandled exception.", exc_info=True
        )
    finally:
        # ------- Disable synchronous mode and clean up world -------
        if SYNCHRONOUS:
            world.apply_settings(original_settings)
            SYNCHRONOUS = False
            LOGGER.debug("Synchronous mode disabled.")
            server_fps = "Variable"

        # Clean up all actors
        world_actor_cleanup(world)

        # Remove ego vehicle and sensors from tracking lists
        VEHICLE_LIST.remove(EGO_VEHICLE)
        SENSOR_LIST.remove(EGO_COLLISION_SENSOR.sensor)
        SENSOR_LIST.remove(EGO_GNSS_SENSOR.sensor)
        SENSOR_LIST.remove(EGO_IMU_SENSOR.sensor)
        SENSOR_LIST.remove(EGO_LANE_INVASION_SENSOR.sensor)

        time.sleep(0.5)
        LOGGER.info("Simulation stopped.\n")


# ----------------- Entry point -----------------
if __name__ == "__main__":
    # Configure global logging
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime}.{msecs:03.0f} | {levelname:<8} | {name:<13} | {message}",
        style="{",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],  # Output to console
        force=True,  # Override any existing logging setup
    )
    LOGGER = logging.getLogger(__name__)

    # Run CARLA baseline path-tracking scenario
    main()
