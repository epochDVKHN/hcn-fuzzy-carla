import carla


def main():
    """
    Connects to CARLA, loads Town01, and prints its default weather parameters.
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = None

    try:
        # --- 1. Load the specific map you want to inspect ---
        print("Loading Town01...")
        world = client.load_world("Town01_Opt")
        # A small tick to ensure the world is ready
        world.tick()
        print("Map loaded.")

        # --- 2. Get the weather object from the world ---
        weather = world.get_weather()
        print("\n-------------------------------------------")
        print("Default Weather Parameters for Town01:")
        print("-------------------------------------------")

        # --- 3. Print the parameters ---
        # The object has a handy __str__ method for a quick summary
        print(weather)

        # You can also access individual parameters like this:
        # print(f"\nIndividual Parameters:")
        # print(f"  Cloudiness: {weather.cloudiness}")
        # print(f"  Precipitation: {weather.precipitation}")
        # print(f"  Puddles: {weather.precipitation_deposits}")
        # print(f"  Sun Altitude: {weather.sun_altitude_angle}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Optional: Reload the default map if you changed it
        # client.load_world('Town10HD')
        print("\nScript finished.")


if __name__ == "__main__":
    main()
