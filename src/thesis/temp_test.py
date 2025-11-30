import logging
import sys
import time

# Add CARLA agents path
sys.path.append(
    r"C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor\PythonAPI\carla"
)

import carla
import numpy as np
import skfuzzy as fuzz
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import compute_distance, get_speed
from skfuzzy import control as ctrl


class FuzzyThrottleBrakeController:
    def __init__(self, destination_distance: float):
        max_distance = int(round(destination_distance, 0)) + 1
        # Define antecedent
        self.target_distance = ctrl.Antecedent(
            universe=np.arange(0, max_distance, 1), label="target_distance"
        )
        # Define consequent
        self.throttle_brake = ctrl.Consequent(
            universe=np.arange(0.0, 1.01, 0.01),
            label="throttle_brake",
            defuzzify_method="centroid",
        )
        # Define antecedent membership functions
        self.target_distance["close"] = fuzz.trimf(
            x=self.target_distance.universe,
            abc=[0, 0, max_distance * 0.05],
        )
        self.target_distance["medium"] = fuzz.trapmf(
            x=self.target_distance.universe,
            abcd=[
                max_distance * 0.03,
                max_distance * 0.15,
                max_distance * 0.6,
                max_distance * 0.75,
            ],
        )
        self.target_distance["far"] = fuzz.trimf(
            x=self.target_distance.universe,
            abc=[max_distance * 0.7, max_distance, max_distance],
        )
        # Define consequent membership functions
        self.throttle_brake["brake"] = fuzz.trimf(
            x=self.throttle_brake.universe,
            abc=[0.0, 0.1, 0.35],
        )
        self.throttle_brake["maintain_speed"] = fuzz.trapmf(
            x=self.throttle_brake.universe,
            abcd=[0.3, 0.4, 0.6, 0.7],
        )
        self.throttle_brake["accelerate"] = fuzz.trimf(
            x=self.throttle_brake.universe,
            abc=[0.65, 1, 1],
        )
        # Define fuzzy rules
        rule_far = ctrl.Rule(
            antecedent=self.target_distance["far"],
            consequent=(self.throttle_brake["accelerate"] % 1.0),
            label="rule_far",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        rule_nearby = ctrl.Rule(
            antecedent=self.target_distance["medium"],
            consequent=(self.throttle_brake["maintain_speed"] % 1.0),
            label="rule_nearby",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        rule_close = ctrl.Rule(
            antecedent=self.target_distance["close"],
            consequent=(self.throttle_brake["brake"] % 1.0),
            label="rule_close",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Assemble the fuzzy control system
        self.control_system = ctrl.ControlSystem(
            rules=[rule_far, rule_nearby, rule_close]
        )
        self.control_simulation = ctrl.ControlSystemSimulation(
            control_system=self.control_system,
            clip_to_bounds=True,
            cache=True,
            flush_after_run=1000,
            lenient=True,
        )

    # Execute fuzzy control system
    def compute_vehicle_control(self, distance_to_destination: float) -> float:
        self.control_simulation.inputs(
            {
                "target_distance": distance_to_destination,
            }
        )
        self.control_simulation.compute()
        return self.control_simulation.output["throttle_brake"]


def actor_cleanup(world: carla.World) -> None:
    def safe_destroy(actor: carla.Actor) -> None:
        if actor.is_alive:
            try:
                actor.destroy()
            except RuntimeError as e:
                logging.warning(
                    f"Handled error destroying Actor ID={actor.id} "
                    f"Type={actor.type_id}: {e}"
                )

    # Get all the actors from the world
    all_actors = world.get_actors()

    # Filter corresponding actor blueprints
    actor_controller = all_actors.filter("controller.*")
    actor_sensor = all_actors.filter("sensor.*")
    actor_static = all_actors.filter("static.*")
    actor_vehicle = all_actors.filter("vehicle.*")
    actor_walker = all_actors.filter("walker.*")

    # Delete corresponding actor blueprints
    if actor_controller:
        logging.debug(f"Destroying {len(actor_controller)} WalkerAIControllers.")
        for controller in actor_controller:
            if controller.is_alive:
                controller.stop()
                safe_destroy(controller)

    if actor_sensor:
        logging.debug(f"Destroying {len(actor_sensor)} Sensors.")
        for sensor in actor_sensor:
            if sensor.is_alive:
                if sensor.is_listening():
                    sensor.stop()
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


def get_route_length(agent: BasicAgent) -> float:
    route_length = 0.0
    path = agent.get_local_planner().get_plan()

    # Iterate through the path and cumulatively sum the Euclidean distances between consecutive point pairs
    for i in range(len(path) - 1):
        point1 = path[i][0].transform.location
        point2 = path[i + 1][0].transform.location
        route_length += compute_distance(point1, point2)

    return route_length


def main():
    """_summary_
    The main function to run the CARLA simulation with the fuzzy controller.
    """
    SYNCHRONOUS = True
    HOST = "localhost"
    PORT = 2000
    EGO_ACTIVE = False
    DISPLAY_TELEMETRY = False
    VEHICLE_LIST = []
    ORIGIN_DRAW_HEIGHT = 1
    DESTINATION_DRAW_HEIGHT = 1

    # Set up ego vehicle lights
    EGO_LIGHT_MASK = carla.VehicleLightState.Position

    # Indices specific to Town01_Opt only
    EGO_START_INDEX = 182
    EGO_STOP_INDEX = 186

    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)
        world = client.get_world()

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        settings = world.get_settings()
        spectator = world.get_spectator()

        # Enable synchronous mode
        delta_seconds = 0.05  # 20 FPS
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)
        SYNCHRONOUS = True
        logging.debug("Synchronous mode enabled.")
        logging.info("Simulation started.")

        # Get spawn points
        spawn_points = map.get_spawn_points()

        # Draw the spawn point locations as numbers in the map
        # for i, spawn_point in enumerate(spawn_points):
        #     world.debug.draw_string(
        #         spawn_point.location,
        #         str(i),
        #         life_time=120,
        #         color=carla.Color(255, 0, 0),
        #     )

        # Set up ego navigation endpoints
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
        EGO_VEHICLE.set_light_state(EGO_LIGHT_MASK)
        EGO_ACTIVE = True
        VEHICLE_LIST.append(EGO_VEHICLE)
        logging.info(f"Spawned Ego {EGO_VEHICLE.type_id} at Origin {EGO_ORIGIN}.")

        # Make the server wait until the environment stabilizes
        for _ in range(1):
            world.tick()

        # Set up ego basic agent
        # The basic agent handles path planning and lateral control while the fuzzy controller handles longitudinal control
        EGO_AGENT = BasicAgent(
            EGO_VEHICLE,
            target_speed=50,
            opt_dict={
                "ignore_traffic_lights": True,
                "ignore_stop_signs": True,
                "max_brake": 1.0,
                "max_throttle": 1.0,
                "follow_speed_limits": False,
            },
        )
        logging.debug("Ego Basic Agent initialized.")

        EGO_AGENT.set_destination(EGO_DESTINATION)
        logging.info(f"Ego Basic Agent set to Destination {EGO_DESTINATION}.")

        # Make the server wait until the environment stabilizes
        for _ in range(5):
            world.tick()

        # Display ego vehicle telemetry
        EGO_VEHICLE.show_debug_telemetry(DISPLAY_TELEMETRY)

        # Calculate the total path length
        origin_destination_route_distance = get_route_length(EGO_AGENT)

        # Initialize fuzzy controller
        EGO_FUZZY_CONTROLLER = FuzzyThrottleBrakeController(
            destination_distance=origin_destination_route_distance
        )
        logging.debug("Ego Fuzzy Logic Controller initialized.")

        # Make the server wait until the environment stabilizes
        for _ in range(1):
            world.tick()

        # Draw an agent's waypoints along the planned route to be taken between the origin and the destination
        # EGO_LOCAL_PLANNER = EGO_AGENT.get_local_planner()
        # EGO_PLAN = EGO_LOCAL_PLANNER.get_plan()
        # for i, (plan_point, _) in enumerate(EGO_PLAN):
        #     world.debug.draw_string(
        #         plan_point.transform.location,
        #         str(i),
        #         draw_shadow=True,
        #         color=carla.Color(255, 0, 0),
        #         life_time=60,
        #     )

        # Draw origin and destination waypoints
        ORIGIN_DRAW = EGO_ORIGIN + carla.Location(z=ORIGIN_DRAW_HEIGHT)
        DESTINATION_DRAW = EGO_DESTINATION + carla.Location(z=DESTINATION_DRAW_HEIGHT)
        world.debug.draw_string(
            ORIGIN_DRAW,
            "Start",
            draw_shadow=True,
            color=carla.Color(255, 0, 0),
            life_time=60,
        )
        world.debug.draw_string(
            DESTINATION_DRAW,
            "Stop",
            draw_shadow=True,
            color=carla.Color(255, 0, 0),
            life_time=60,
        )

        # Game loop
        log_counter = 0
        EGO_BRAKE = 0.3
        SPEED_FLAG = False
        while True:
            world.tick()
            if EGO_ACTIVE:
                EGO_TRANSFORM = EGO_VEHICLE.get_transform()

                # Set up spectator camera
                # For chase camera view
                spectator_transform = carla.Transform(
                    EGO_TRANSFORM.transform(carla.Location(x=-4, z=2)),
                    EGO_TRANSFORM.rotation,
                )
                spectator.set_transform(spectator_transform)

                # Run the basic agent with the fuzzy longitudinal controls
                EGO_CONTROL = EGO_AGENT.run_step()
                EGO_CONTROL_STEER = EGO_CONTROL.steer
                # Compute the distance between current position and destination
                # goal_distance = compute_distance(
                #     location_1=EGO_TRANSFORM.location, location_2=EGO_DESTINATION
                # )
                goal_distance = get_route_length(EGO_AGENT)
                # Compute the fuzzy throttle and brake control outputs
                throttle_brake_output = EGO_FUZZY_CONTROLLER.compute_vehicle_control(
                    distance_to_destination=goal_distance
                )
                # Combine and apply the throttle and brake controls
                # Take the positive output, else 0
                EGO_CONTROL_THROTTLE = max(0, throttle_brake_output)
                # Take the negated output, else 0
                EGO_CONTROL_BRAKE = max(0, -throttle_brake_output)
                EGO_FUZZY_CONTROL = carla.VehicleControl(
                    throttle=0.0 if SPEED_FLAG else EGO_CONTROL_THROTTLE,
                    steer=EGO_CONTROL_STEER,
                    brake=EGO_BRAKE if SPEED_FLAG else EGO_CONTROL_BRAKE,
                )

                # if get_speed(EGO_VEHICLE) >= 50.0:
                #     SPEED_FLAG = True
                # EGO_FUZZY_CONTROL = carla.VehicleControl(
                #     throttle=0.0,
                #     steer=EGO_CONTROL_STEER,
                #     brake=EGO_BRAKE,
                # )

                EGO_VEHICLE.apply_control(EGO_FUZZY_CONTROL)

                if log_counter % 10 == 0:
                    logging.info(
                        f"Distance to Destination: {goal_distance:<6.2f}m | "
                        f"Agent Steer: {EGO_CONTROL_STEER:<5.2f} | "
                        f"Agent Velocity: {get_speed(EGO_VEHICLE):<6.2f} km/h -> T: {EGO_FUZZY_CONTROL.throttle:.2f}, B: {EGO_FUZZY_CONTROL.brake:.2f}"
                    )

            # Terminate the game loop
            if EGO_ACTIVE and EGO_AGENT.done():
                logging.info("Ego Basic Agent reached destination.")
                logging.info("Simulation terminated.")
                time.sleep(0.017)
                VEHICLE_LIST.remove(EGO_VEHICLE)
                EGO_ACTIVE = False
                break

            log_counter += 1
            time.sleep(0.017)

    except KeyboardInterrupt:
        logging.warning("Simulation interrupted by user!")
    except Exception as _:
        logging.critical("An unhandled exception occurred!", exc_info=True)
    finally:
        # Disable synchronous mode
        if SYNCHRONOUS:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            SYNCHRONOUS = False
            logging.debug("Synchronous mode disabled.")
        # Stop and destroy all actors in the world
        actor_cleanup(world)
        time.sleep(0.5)
        logging.info("Done.\n")


if __name__ == "__main__":
    # Set up logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} | {levelname:<8} | {name} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Run game loop
    main()
