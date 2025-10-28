import logging
import sys

import carla

# Add CARLA agents path
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import compute_distance


def actor_cleanup(world: carla.World) -> None:
    """
    Destroy all active actors in the CARLA world in a safe and controlled manner.

    This function retrieves all actors currently present in the world and
    attempts to destroy them by category. It ensures proper shutdown of
    controllers and sensors before destruction to avoid runtime conflicts
    or dangling references.

    Categories handled:
        - controller.*  : Walker AI controllers (stopped before destruction)
        - sensor.*      : Sensors (stopped if actively listening)
        - static.*      : Static props or decorations
        - vehicle.*     : Vehicles (all types)
        - walker.*      : Pedestrians

    :param world: The CARLA world instance from which actors will be destroyed.
    :type world: carla.World
    :return: None
    """

    def safe_destroy(actor: carla.Actor) -> None:
        """
        Attempt to safely destroy an actor while handling common runtime errors.

        This function checks if the actor is still alive and then tries to destroy it.
        If a RuntimeError occurs (e.g., the actor is being destroyed concurrently
        or still in use by another process), the exception is caught and logged
        without interrupting the cleanup process.

        :param actor: The CARLA actor instance to destroy.
        :type actor: carla.Actor
        :return: None
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
    actor_static = all_actors.filter("static.*")
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

    if actor_static:
        logging.debug(f"Destroying {len(actor_static)} Props.")
        for static in actor_static:
            if static.is_alive:
                safe_destroy(static)

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

    logging.info("World cleanup completed. All actors destroyed.")


def get_route_length(agent: BasicAgent) -> float:
    """
    Compute the total length of the planned route for a given agent.

    The route is obtained from the agent's local planner and is expected to be a sequence
    of (Waypoint, RoadOption) pairs. The total length is calculated by summing the
    Euclidean distances between consecutive waypoint locations along the route.

    :param agent: The CARLA BasicAgent whose planned route length will be computed.
    :type agent: BasicAgent
    :return: Total route length in meters.
    :rtype: float
    """
    route_length = 0.0
    path = (
        agent.get_local_planner().get_plan()
    )  # Retrieve the planned route as a list of (Waypoint, RoadOption) pairs

    # Accumulate Euclidean distances between consecutive waypoints
    for i in range(len(path) - 1):
        point1 = path[i][0].transform.location
        point2 = path[i + 1][0].transform.location
        route_length += compute_distance(point1, point2)

    return route_length
