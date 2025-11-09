import enum
import logging
import math
import sys
from typing import Deque, List, Tuple

import carla

# NOTE: UPDATE the vehicle_plan below to YOUR CARLA 0.9.15 installation folder containing 'PythonAPI\carla'.
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import compute_distance


class SpectatorCameraPositions(enum.Enum):
    """
    Enumerates available spectator camera perspectives.

    Each value corresponds to a predefined camera position
    used for switching between different viewing angles in simulation.
    """

    LOW_ANGLE_CHASE_VIEW = (
        1  # Low, cinematic chase view positioned tightly behind the vehicle
    )
    FRONT_BUMPER_VIEW = 2  # First-person view from the front bumper
    REVERSE_TRACK_VIEW = (
        3  # Forward-facing shot from ahead of the vehicle, moving backward
    )
    REAR_BUMPER_VIEW = 4  # Rear-facing view fixed just behind the vehicle’s bumper
    HIGH_ANGLE_CHASE_VIEW = (
        5  # Elevated chase view showing the car, lane lines, and road ahead
    )
    FRONT_QUARTER_VIEW = (
        6  # Angled front-quarter view capturing a dynamic three-quarter perspective
    )
    BIRD_EYE_VIEW = 7  # Top-down drone-style view directly above the vehicle


def world_snapshot_tick(world: carla.World, tick_freq: int = 1) -> carla.WorldSnapshot:
    """
    Advance the CARLA simulation by a specified number of ticks and return the latest snapshot.

    Args:
        world (carla.World): The CARLA world instance.
        tick_freq (int, optional): Number of simulation ticks to advance. Defaults to 1.

    Returns:
        carla.WorldSnapshot: Snapshot of the world after advancing the simulation.

    Notes:
        - In synchronous mode, the simulation advances immediately on world.tick().
        - In asynchronous mode, the simulation waits for the next update via world.wait_for_tick().
    """
    world_settings = world.get_settings()

    for _ in range(tick_freq):
        if world_settings.synchronous_mode:
            # Advance the simulation immediately (synchronous mode)
            world.tick()
            world_snapshot = world.get_snapshot()
        else:
            # Wait for the next simulation update (asynchronous mode)
            world_snapshot = world.wait_for_tick()

    return world_snapshot


def world_actor_cleanup(world: carla.World) -> None:
    """
    Safely destroy all active actors in the CARLA world.

    Args:
        world (carla.World): The CARLA world containing the actors.

    Returns:
        None

    Notes:
        Actors are destroyed by category to ensure proper shutdown of controllers and sensors before removal, avoiding runtime errors or dangling references.

        Handled categories:
        - controller.* : Walker AI controllers (stopped before destruction)
        - sensor.*     : Sensors (stopped if listening)
        - static.*     : Static props
        - vehicle.*    : Vehicles
        - walker.*     : Pedestrians
    """

    def safe_destroy(actor: carla.Actor) -> None:
        """
        Safely destroy an actor while handling runtime errors.

        Args:
            actor (carla.Actor): The actor to destroy.

        Returns:
            None
        """
        if actor.is_alive:
            try:
                actor.destroy()
            except RuntimeError as e:
                logging.warning(
                    f"Handled error destroying Actor ID={actor.id} "
                    f"Type={actor.type_id}: {e}"
                )

    # Retrieve all actor instances currently present in the world
    all_actors = world.get_actors()

    # Categorize actors by type for structured cleanup
    actor_controller = all_actors.filter("controller.*")
    actor_sensor = all_actors.filter("sensor.*")
    # actor_static = all_actors.filter("static.*")
    actor_vehicle = all_actors.filter("vehicle.*")
    actor_walker = all_actors.filter("walker.*")

    # Destroy actors category by category, ensuring proper shutdown where necessary
    if actor_controller:
        logging.debug(f"Destroying {len(actor_controller)} WalkerAIControllers.")
        for controller in actor_controller:
            if controller.is_alive:
                controller.stop()  # Stop controller before destruction
                safe_destroy(controller)

        # Wait briefly to ensure the world state stabilizes
        world_snapshot_tick(world)

    if actor_sensor:
        logging.debug(f"Destroying {len(actor_sensor)} Sensors.")
        for sensor in actor_sensor:
            if sensor.is_alive:
                if sensor.is_listening():
                    sensor.stop()  # Stop active sensors before destruction
                safe_destroy(sensor)

        # Wait briefly to ensure the world state stabilizes
        world_snapshot_tick(world)

    # Uncomment to remove all static props (including default CARLA map assets); this affects visuals only.
    # if actor_static:
    #     logging.debug(f"Destroying {len(actor_static)} Props.")
    #     for static in actor_static:
    #         if static.is_alive:
    #             safe_destroy(static)

    #     # Wait briefly to ensure the world state stabilizes
    #     world_tick_snapshot(world)

    if actor_vehicle:
        logging.debug(f"Destroying {len(actor_vehicle)} Vehicles.")
        for vehicle in actor_vehicle:
            if vehicle.is_alive:
                safe_destroy(vehicle)

        # Wait briefly to ensure the world state stabilizes
        world_snapshot_tick(world)

    if actor_walker:
        logging.debug(f"Destroying {len(actor_walker)} Walkers.")
        for walker in actor_walker:
            if walker.is_alive:
                safe_destroy(walker)

        # Wait briefly to ensure the world state stabilizes
        world_snapshot_tick(world)


def get_route_length(vehicle_agent: BasicAgent) -> float:
    """
    Calculate the total length of an agent's planned route.

    The route is obtained from the agent's local planner as a list of
    (Waypoint, RoadOption) pairs. The length is computed by summing the
    Euclidean distances between consecutive waypoint locations.

    Args:
        agent (BasicAgent): The CARLA agent whose route length is calculated.

    Returns:
        float: Total route length in meters.
    """
    route_length = 0.0
    # List of [(Waypoint, RoadOption), ...]
    vehicle_plan = vehicle_agent.get_local_planner().get_plan()

    # Sum Euclidean distances between consecutive waypoints
    for i in range(len(vehicle_plan) - 1):
        waypoint1 = vehicle_plan[i][0].transform.location
        waypoint2 = vehicle_plan[i + 1][0].transform.location
        route_length += compute_distance(waypoint1, waypoint2)

    return route_length


def draw_spawn_waypoints(
    world: carla.World,
    spawn_waypoints: List[carla.Transform],
    display_time: float = 120,
) -> None:
    """
    Draws labeled spawn waypoints in the CARLA world.

    Each spawn waypoint is annotated with its index in red text above its location.

    Args:
        world (carla.World): The CARLA simulation world instance.
        spawn_waypoints (List[carla.Transform]): List of spawn waypoint transforms.
        display_time (float, optional): Duration (in seconds) for which labels remain visible. Defaults to 120.

    Returns:
        None

    """
    for i, spawn_waypoint in enumerate(spawn_waypoints):
        # Draw index label above each spawn waypoint
        world.debug.draw_string(
            location=spawn_waypoint.location,
            text=str(i),
            draw_shadow=True,
            color=carla.Color(r=255, g=0, b=0),
            life_time=display_time,
        )


def set_vehicle_lights(
    vehicle_default_lights: int,
    vehicle_control: carla.VehicleControl,
    vehicle_agent: BasicAgent,
    vehicle_step_history: Deque[Tuple[carla.Waypoint, RoadOption]],
) -> carla.VehicleLightState:
    """
    Update a vehicle's light state based on control input and route intent.

    Args:
        vehicle_default_lights (int): Current default vehicle light state.
        vehicle_control (carla.VehicleControl): Current control signals (throttle, brake, reverse, etc.).
        vehicle_agent (BasicAgent): Agent controlling the vehicle.
        vehicle_step_history (Deque[Tuple[carla.Waypoint, RoadOption]]):
            Recent waypoint and maneuver history from the local planner.

    Returns:
        carla.VehicleLightState: Updated light state including brake, reverse, and turn indicators.
    """
    vehicle_local_planner = vehicle_agent.get_local_planner()
    vehicle_updated_lights = vehicle_default_lights

    # Flags to track upcoming turn intent
    intent_to_turn_left = False
    intent_to_turn_right = False
    LOOK_AHEAD_waypointS = 12  # Number of future waypoints to analyze for turn intent

    # Record current local planner step for continuity
    vehicle_step_history.append(
        vehicle_local_planner.get_incoming_waypoint_and_direction(steps=0)
    )

    # Longitudinal lights: activate brake and reverse lights based on control input
    if vehicle_control.brake > 0.0:
        vehicle_updated_lights |= carla.VehicleLightState.Brake
    if vehicle_control.reverse:
        vehicle_updated_lights |= carla.VehicleLightState.Reverse

    # Get recent and upcoming waypoints for turn intent detection
    vehicle_history_waypoints = list(vehicle_step_history)
    vehicle_future_waypoints = list(vehicle_local_planner.get_plan())[
        1:LOOK_AHEAD_waypointS
    ]
    vehicle_road_plan = vehicle_history_waypoints + vehicle_future_waypoints

    # Detect turn-related road options in recent and future steps
    for _, roadoption in vehicle_road_plan:
        if roadoption in (RoadOption.LEFT, RoadOption.CHANGELANELEFT):
            intent_to_turn_left = True
            break
        if roadoption in (RoadOption.RIGHT, RoadOption.CHANGELANERIGHT):
            intent_to_turn_right = True
            break

    # Lateral lights: activate corresponding blinkers based on detected turn intent
    if intent_to_turn_right:
        vehicle_updated_lights |= carla.VehicleLightState.RightBlinker
    if intent_to_turn_left:
        vehicle_updated_lights |= carla.VehicleLightState.LeftBlinker

    return carla.VehicleLightState(vehicle_updated_lights)


def set_spectator_camera(
    vehicle: carla.Vehicle,
    camera_position: SpectatorCameraPositions,
) -> carla.Transform:
    """
    Position the spectator camera relative to a vehicle based on a predefined view.

    Args:
        vehicle (carla.Vehicle): The target vehicle to align the camera with.
        camera_position (SpectatorCameraPositions): The desired camera viewpoint preset.

    Returns:
        carla.Transform: The computed world-space transform for the spectator camera.

    Raises:
        TypeError: If `camera_position` is not a SpectatorCameraPositions enum value.
    """
    if not isinstance(camera_position, SpectatorCameraPositions):
        raise TypeError(
            f"camera_position must be an instance of SpectatorCameraPositions enum, not {type(camera_position).__name__}."
        )

    # Get the current world transform of the vehicle
    ego_transform = vehicle.get_transform()

    # Default neutral offsets
    relative_location = carla.Location(x=0, y=0, z=0)
    relative_rotation = carla.Rotation(pitch=0, yaw=0, roll=0)

    # Assign relative offsets and rotations based on the selected view
    if camera_position == SpectatorCameraPositions.LOW_ANGLE_CHASE_VIEW:
        relative_location = carla.Location(x=-4.3, y=0.0, z=2.2)
        relative_rotation = carla.Rotation(pitch=2.0)

    elif camera_position == SpectatorCameraPositions.FRONT_BUMPER_VIEW:
        relative_location = carla.Location(x=1.99, y=0.0, z=1.3)
        relative_rotation = carla.Rotation(pitch=0.0)

    elif camera_position == SpectatorCameraPositions.REVERSE_TRACK_VIEW:
        relative_location = carla.Location(x=5.76, y=1.5, z=1.7)
        relative_rotation = carla.Rotation(pitch=2.0)

    elif camera_position == SpectatorCameraPositions.REAR_BUMPER_VIEW:
        relative_location = carla.Location(x=-1.34, y=0.0, z=1.3)
        relative_rotation = carla.Rotation(pitch=0.0)

    elif camera_position == SpectatorCameraPositions.HIGH_ANGLE_CHASE_VIEW:
        relative_location = carla.Location(x=-4.37, y=0.0, z=3.9)
        relative_rotation = carla.Rotation(pitch=-16.3)

    elif camera_position == SpectatorCameraPositions.FRONT_QUARTER_VIEW:
        relative_location = carla.Location(x=0.32, y=-1.4, z=1.3)
        relative_rotation = carla.Rotation(pitch=0.0)

    elif camera_position == SpectatorCameraPositions.BIRD_EYE_VIEW:
        relative_location = carla.Location(x=2, y=0, z=29.0)
        relative_rotation = carla.Rotation(pitch=-90.0)

    # Adjust yaw for direction-dependent views
    if camera_position in (
        SpectatorCameraPositions.REVERSE_TRACK_VIEW,
        SpectatorCameraPositions.REAR_BUMPER_VIEW,
    ):
        adjusted_yaw = ego_transform.rotation.yaw + 180.0  # Flip to face rearward
    elif camera_position == SpectatorCameraPositions.FRONT_QUARTER_VIEW:
        adjusted_yaw = ego_transform.rotation.yaw + 25.0  # Slight angled offset
    elif camera_position == SpectatorCameraPositions.BIRD_EYE_VIEW:
        adjusted_yaw = -180.0  # Static top-down orientation
    else:
        adjusted_yaw = ego_transform.rotation.yaw  # Default follows vehicle heading

    # Transform local camera offset relative to the vehicle's position
    world_location = ego_transform.transform(relative_location)
    world_rotation = carla.Rotation(
        pitch=relative_rotation.pitch,
        yaw=adjusted_yaw,
        roll=relative_rotation.roll,
    )

    return carla.Transform(world_location, world_rotation)


def draw_plan_waypoints(
    world: carla.World,
    vehicle_agent: BasicAgent,
    frame_num: int,
    display_freq: int = 10,
    display_nearest_points: int = 30,
    display_height: float = 0.5,
    display_time: float = 1.0,
) -> None:
    """
    Draws the nearest planned waypoints from a CARLA vehicle agent.

    Visualizes the next few waypoints in the agent’s local plan at a set
    frame interval using the CARLA `world.debug` drawing API.

    Args:
        world (carla.World): The CARLA simulation world.
        vehicle_agent (BasicAgent): Agent providing access to the local planner.
        frame_num (int): Current simulation frame number.
        display_freq (int, optional): Draw frequency (every N frames). Defaults to 10.
        display_nearest_points (int, optional): Number of waypoints to draw. Defaults to 30.
        display_height (float, optional): Height offset for visualization. Defaults to 0.5.
        display_time (float, optional): Duration (s) each waypoint is displayed. Defaults to 1.0.

    Returns:
        None
    """
    # Only draw every `display_freq` frames to reduce overhead
    if frame_num % display_freq != 0:
        return

    # Retrieve the nearest waypoints from the agent’s local plan
    vehicle_plan = list(vehicle_agent.get_local_planner().get_plan())[
        :display_nearest_points
    ]

    for waypoint, _ in vehicle_plan:
        # Compute start (base) and end (directional) points of the waypoint arrow
        waypoint_transform = waypoint.transform
        begin_waypoint = waypoint_transform.location + carla.Location(z=display_height)
        waypoint_angle = math.radians(waypoint_transform.rotation.yaw)
        end_waypoint = begin_waypoint + carla.Location(
            x=math.cos(waypoint_angle), y=math.sin(waypoint_angle)
        )

        # Draw arrow representing the waypoint direction
        world.debug.draw_arrow(
            begin_waypoint,
            end_waypoint,
            thickness=0.1,
            arrow_size=0.3,
            color=carla.Color(r=0, g=1, b=1),  # Cyan color
            life_time=display_time,
        )


def generate_custom_route(
    route_indices: List[int],
    spawn_points: List[carla.Transform],
    map: carla.Map,
    sampling_resolution: float = 2.0,
) -> List[Tuple[carla.Waypoint, RoadOption]]:
    """
    Generate a continuous route between selected spawn points.

    This function constructs a route by connecting specified spawn points in sequence using CARLA's GlobalRoutePlanner.

    Args:
        route_indices (List[int]): Indices of spawn points to include in the route.
        spawn_points (List[carla.Transform]): Available spawn point transforms in the map.
        map (carla.Map): CARLA map instance used for route planning.
        sampling_resolution (float, optional): Distance between sampled waypoints (meters). Defaults to 2.0.

    Returns:
        List[Tuple[carla.Waypoint, RoadOption]]: Ordered list of (waypoint, road option) tuples forming the full route.
    """
    # Extract locations for the specified spawn points
    route_locations = [spawn_points[i].location for i in route_indices]

    # Initialize the route planner
    global_route_planner = GlobalRoutePlanner(map, sampling_resolution)
    route_plan = []

    # Build the full route by linking each consecutive pair of spawn points
    for i in range(len(route_locations) - 1):
        start_location = route_locations[i]
        end_location = route_locations[i + 1]

        # Generate the route segment between the two locations
        route_segment = global_route_planner.trace_route(start_location, end_location)

        # Avoid duplicate waypoints by skipping the first waypoint of subsequent segments
        if i > 0:
            route_plan.extend(route_segment[1:])
        else:
            route_plan.extend(route_segment)

    return route_plan


def get_actor_display_name(actor: carla.Actor, truncate: int = 250) -> str:
    """
    Generate a concise, human-readable name for a CARLA actor.

    Args:
        actor (carla.Actor): The CARLA actor whose display name is required.
        truncate (int, optional): Maximum allowed length of the name. Defaults to 250.

    Returns:
        str: Formatted actor name, truncated with an ellipsis if it exceeds the limit.
    """
    type_id = actor.type_id
    type_parts = type_id.split(".")

    if len(type_parts) > 2 and type_parts[0] == "sensor" and type_parts[1] == "other":
        # Handle 'sensor.other.*' separately (e.g., 'sensor.other.lane_invasion' → 'Lane Invasion')
        display_name = type_parts[-1].replace("_", " ").title()
    else:
        # Handle general actor types (e.g., 'vehicle.tesla.model3' → 'Tesla Model3')
        display_name = " ".join(type_parts[1:]).replace("_", " ").title()

    # Truncate and append an ellipsis if the name exceeds the specified length
    return (
        (display_name[: truncate - 1] + "\u2026")
        if len(display_name) > truncate
        else display_name
    )
