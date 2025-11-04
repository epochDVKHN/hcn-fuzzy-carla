import enum
import logging
import sys
from typing import Deque, List, Tuple

import carla

# NOTE: UPDATE the path below to YOUR CARLA 0.9.15 installation folder containing 'PythonAPI\carla'.
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)

from agents.navigation.basic_agent import BasicAgent
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


def actor_cleanup(world: carla.World) -> None:
    """
    Safely destroy all active actors in the CARLA world.

    Actors are destroyed by category to ensure proper shutdown of controllers
    and sensors before removal, avoiding runtime errors or dangling references.

    Handled categories:
        - controller.* : Walker AI controllers (stopped before destruction)
        - sensor.*     : Sensors (stopped if listening)
        - static.*     : Static props
        - vehicle.*    : Vehicles
        - walker.*     : Pedestrians

    Args:
        world (carla.World): The CARLA world containing the actors.

    Returns:
        None
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

    if actor_sensor:
        logging.debug(f"Destroying {len(actor_sensor)} Sensors.")
        for sensor in actor_sensor:
            if sensor.is_alive:
                if sensor.is_listening():
                    sensor.stop()  # Stop active sensors before destruction
                safe_destroy(sensor)

    # Uncomment to remove all static props (including default CARLA map assets); this affects visuals only.
    # if actor_static:
    #     logging.debug(f"Destroying {len(actor_static)} Props.")
    #     for static in actor_static:
    #         if static.is_alive:
    #             safe_destroy(static)

    if actor_vehicle:
        logging.debug(f"Destroying {len(actor_vehicle)} Vehicles.")
        for vehicle in actor_vehicle:
            if vehicle.is_alive:
                safe_destroy(vehicle)

    if actor_walker:
        logging.debug(f"Destroying {len(actor_walker)} Walkers.")
        for walker in actor_walker:
            if walker.is_alive:
                safe_destroy(walker)


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
    path = vehicle_agent.get_local_planner().get_plan()  # [(Waypoint, RoadOption), ...]

    # Sum Euclidean distances between consecutive waypoints
    for i in range(len(path) - 1):
        point1 = path[i][0].transform.location
        point2 = path[i + 1][0].transform.location
        route_length += compute_distance(point1, point2)

    return route_length


def draw_spawn_points(
    world: carla.World, spawn_points: List[carla.Transform], flag: bool
) -> None:
    """
    Visualize spawn points in the CARLA world with indexed labels.

    Args:
        world (carla.World): The active CARLA world instance.
        spawn_points (List[carla.Transform]): Spawn point transforms to label.
        flag (bool): Whether to render the labels.

    Returns:
        None
    """
    if not flag:
        return
    for i, spawn_point in enumerate(spawn_points):
        # Draw index label above each spawn point
        world.debug.draw_string(
            location=spawn_point.location,
            text=str(i),
            draw_shadow=True,
            color=carla.Color(r=255, g=0, b=0),
            life_time=120,
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
    LOOK_AHEAD_POINTS = 12  # Number of future waypoints to analyze for turn intent

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
        1:LOOK_AHEAD_POINTS
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
    spectator_camera: carla.Actor,
    vehicle: carla.Vehicle,
    camera_position: SpectatorCameraPositions,
) -> carla.Transform:
    """Position the spectator camera relative to a vehicle based on a predefined view.

    Args:
        spectator_camera (carla.Actor): The spectator camera actor.
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
