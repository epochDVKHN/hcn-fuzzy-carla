import logging
import sys
from typing import List, Tuple

import carla

# NOTE: UPDATE the path below to YOUR CARLA 0.9.15 installation folder containing 'PythonAPI\carla'.
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.tools.misc import compute_distance


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


def get_route_length(agent: BasicAgent) -> float:
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
    path = agent.get_local_planner().get_plan()  # [(Waypoint, RoadOption), ...]

    # Sum Euclidean distances between consecutive waypoints
    for i in range(len(path) - 1):
        point1 = path[i][0].transform.location
        point2 = path[i + 1][0].transform.location
        route_length += compute_distance(point1, point2)

    return route_length


def draw_spawn_points(world: carla.World, spawn_points: List[carla.Transform]) -> None:
    """
    Render spawn points in the CARLA world with their indices displayed.

    Args:
        world (carla.World): The CARLA world instance.
        spawn_points (List[carla.Transform]): Spawn point transforms to visualize.

    Returns:
        None
    """
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
    vehicle_local_planner: LocalPlanner,
    vehicle_history_steps: List[Tuple[carla.Waypoint, RoadOption]],
) -> carla.VehicleLightState:
    vehicle_updated_lights = vehicle_default_lights
    intent_to_turn_left = False
    intent_to_turn_right = False
    LOOK_AHEAD_POINTS = 12  # Number of waypoints to look ahead

    # Longitudinal lights
    if vehicle_control.brake > 0.0:
        vehicle_updated_lights |= carla.VehicleLightState.Brake
    if vehicle_control.reverse:
        vehicle_updated_lights |= carla.VehicleLightState.Reverse

    # Lateral lights
    # Get the next set of upcoming steps from the local planner
    vehicle_upcoming_steps = list(vehicle_local_planner.get_plan())[1:LOOK_AHEAD_POINTS]
    # Get the road plan for the vehicle accounting for the previous and upcoming steps
    vehicle_road_plan = vehicle_history_steps + vehicle_upcoming_steps

    # Check the road options in the upcoming plan  for any intention to turn
    for _, roadoption in vehicle_road_plan:
        if roadoption in (RoadOption.LEFT, RoadOption.CHANGELANELEFT):
            intent_to_turn_left = True
            break
        if roadoption in (RoadOption.RIGHT, RoadOption.CHANGELANERIGHT):
            intent_to_turn_right = True
            break

    if intent_to_turn_right:
        vehicle_updated_lights |= carla.VehicleLightState.RightBlinker
    if intent_to_turn_left:
        vehicle_updated_lights |= carla.VehicleLightState.LeftBlinker

    return carla.VehicleLightState(vehicle_updated_lights)
