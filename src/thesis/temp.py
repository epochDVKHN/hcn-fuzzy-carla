import os
import sys
import time

import carla

# Ensure the 'agents' module from CARLA is in the Python path
# Adjust this path if your 'agents' directory is elsewhere
# This assumes you are running from the 'src' directory
try:
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "CARLA_0.9.15",
            "PythonAPI",
            "carla",
        )
    )
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    print("Error: CARLA 'agents' module not found.")
    print(
        "Please ensure your Python path includes /path/to/CARLA_0.9.15/PythonAPI/carla"
    )
    sys.exit(1)

# Import our helper functions
from thesis.utils.helpers import cleanup_actors, setup_camera, setup_weather


def main():
    """
    Main function to run the path tracking baseline scenario.
    """
    actor_list = []
    original_settings = None
    client = None

    try:
        # --- 1. Connect to CARLA ---
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # --- 2. Load World and Set Up Environment ---
        world = client.load_world("Town01")
        print("Loaded 'Town01'.")

        # Store original settings to restore them later
        original_settings = world.get_settings()

        # Set synchronous mode for reproducible results
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # --- 3. Set Weather ---
        setup_weather(world, "GoldenHour")

        # --- 4. Define Start and End Points ---
        # Start point: On the straight road near the T-junction
        start_transform = carla.Transform(
            carla.Location(x=192.1, y=133.0, z=0.5), carla.Rotation(yaw=180)
        )
        # End point: Further down the same straight road
        end_location = carla.Location(x=331.1, y=133.3, z=0.5)

        # --- 5. Spawn Ego Vehicle ---
        vehicle_bp = blueprint_library.filter("model.tesla.model3")[0]
        vehicle = world.spawn_actor(vehicle_bp, start_transform)
        actor_list.append(vehicle)
        print(f"Spawned ego vehicle (ID: {vehicle.id}) at {start_transform.location}.")

        # --- 6. Set up Camera and Spectator ---
        setup_camera(world, vehicle, actor_list)

        # Move spectator camera to view the start
        spectator = world.get_spectator()
        spectator_transform = carla.Transform(
            start_transform.location + carla.Location(z=20, x=-15),
            carla.Rotation(pitch=-60, yaw=180),
        )
        spectator.set_transform(spectator_transform)

        # Tick once to apply settings and spawn actors
        world.tick()

        # --- 7. Initialize and Configure the BasicAgent ---
        # The BasicAgent contains its own logic for path planning.
        # We just need to give it the vehicle and a target speed.
        agent = BasicAgent(vehicle, target_speed=30.0)  # Target speed in km/h

        # Set the agent's destination. The agent will use its own
        # internal GlobalRoutePlanner to compute the path.
        agent.set_destination(end_location)
        print(f"Initialized BasicAgent. Destination set to {end_location}.")

        # --- 8. Run the Simulation Loop ---
        print("Starting simulation... Agent is running.")
        start_time = time.time()

        while True:
            # Tick the world to advance the simulation
            world.tick()

            # Run one step of the agent's logic
            # This returns the carla.VehicleControl (throttle, steer, brake)
            control = agent.run_step()

            # Apply the control to the vehicle
            vehicle.apply_control(control)

            # Check if the agent has reached its destination
            if agent.done():
                print("Agent has reached the destination.")
                break

        end_time = time.time()
        print(f"Simulation finished. Total time: {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        # --- 9. Cleanup ---
        if original_settings and world:
            cleanup_actors(world, actor_list, original_settings)
        elif world:
            # Fallback if original_settings wasn't captured
            cleanup_actors(world, actor_list, world.get_settings())

        print("Execution complete.")


if __name__ == "__main__":
    main()
