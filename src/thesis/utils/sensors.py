"""_summary_

Sensor wrapper classes for CARLA.
Adapted from: https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py
"""

import logging
import math
import sys
import weakref
from pathlib import Path
from typing import List, Tuple

import carla

# Add project root path for module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import get_actor_display_name  # noqa: E402


class CollisionSensor(object):
    """
    Stateful wrapper around CARLA's collision sensor for retrieving collisions between its parent and other actors.

    Attaches a collision sensor rigidly to a parent actor and records collision events, storing metadata including frame number, timestamp, impulse vector, and other actor information.
    """

    def __init__(self, parent_actor: carla.Actor) -> None:
        """
        Initialize and attach the collision sensor.

        Args:
            parent_actor (carla.Actor): The actor to which this sensor is attached.

        Returns:
            None
        """
        self.sensor = None
        # Each entry: (frame, timestamp, impulse_x, impulse_y, impulse_z, other_actor_id, other_actor_type_id)
        self.history: List[Tuple[int, float, float, float, float, int, str]] = []
        self._parent = parent_actor
        self._logger = logging.getLogger(__name__)

        world = self._parent.get_world()
        collision_blueprint = world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        # Spawn the collision sensor and attach it rigidly to the parent actor
        self.sensor = world.spawn_actor(
            blueprint=collision_blueprint,
            transform=carla.Transform(),
            attach_to=self._parent,
            attachment=carla.AttachmentType.Rigid,
        )

        # Use a weak reference to prevent circular references between sensor and callback
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )
        self._logger.info("Collision detector sensor online and listening for events.")

    def get_collision_history(
        self,
    ) -> List[Tuple[int, float, float, float, float, int, str]]:
        """
        Return a copy of the recorded collision history.

        Each entry contains:`(frame, timestamp, impulse_x, impulse_y, impulse_z, other_actor_id, other_actor_type_id)`.

        Returns:
            List[Tuple[]]: A shallow copy of the collision events history.
        """
        return list(self.history)  # Return a shallow copy to prevent external mutation

    @staticmethod
    def _on_collision(
        weak_self: weakref.ReferenceType, event: carla.CollisionEvent
    ) -> None:
        """
        Handle incoming collision events.

        Extracts relevant data from the event, logs the collision,and updates the sensor's collision history buffer. If the sensor instance no longer exists, the callback exits silently.

        Args:
            weak_self (weakref.ReferenceType): Weak reference to the CollisionSensor instance.
            event (carla.CollisionEvent): The collision event provided by CARLA.

        Returns:
            None
        """
        self = weak_self()
        if not self:
            # Exit if the CollisionSensor instance has been garbage collected
            return

        actor_type = get_actor_display_name(event.other_actor)
        collision_impulse = event.normal_impulse
        collision_intensity = math.sqrt(
            collision_impulse.x**2 + collision_impulse.y**2 + collision_impulse.z**2
        )

        self._logger.warning(
            f"Collision detected at {event.timestamp: .2fs} s "
            f"with: {actor_type} (ID: {event.other_actor.id}), "
            f"Intensity: {collision_intensity: .2f} N·s."
        )

        # Store the event sensor data attributes
        self.history.append(
            (
                event.frame,
                event.timestamp,
                collision_impulse.x,
                collision_impulse.y,
                collision_impulse.z,
                event.other_actor.id,
                event.other_actor.type_id,
            )
        )

        # Maintain a fixed-length history buffer
        if len(self.history) > 4000:
            self.history.pop(0)


class GNSSSensor(object):
    """
    Stateful wrapper around CARLA's GNSS sensor for retrieving the geolocation of the sensor.

    Spawns a GNSS sensor attached to a parent actor, configures its noise parameters for deterministic measurements, and stores the latest readings: latitude, longitude, altitude, simulation frame, and timestamp.
    """

    def __init__(self, parent_actor: carla.Actor) -> None:
        """
        Initialize a GNSS sensor attached to a CARLA actor.

        Configures the sensor's noise parameters to zero for deterministic readings and spawns it in the simulation world.

        Args:
            parent_actor (carla.Actor): Actor to attach the GNSS sensor to.

        Returns:
            None
        """
        self.sensor = None
        self._parent = parent_actor
        self._logger = logging.getLogger(__name__)

        # Initialize sensor data attributes
        self.lat: float = 0.0
        self.lon: float = 0.0
        self.alt: float = 0.0
        self.frame: int = 0
        self.timestamp: float = 0.0

        world = self._parent.get_world()
        gnss_blueprint = world.get_blueprint_library().find("sensor.other.gnss")

        # Configure GNSS blueprint attributes for deterministic output
        if gnss_blueprint.has_attribute("noise_alt_bias"):
            gnss_blueprint.set_attribute("noise_alt_bias", "0.0")  # Altitude bias mean
        if gnss_blueprint.has_attribute("noise_alt_stddev"):
            gnss_blueprint.set_attribute(
                "noise_alt_stddev", "0.0"
            )  # Altitude standard deviation
        if gnss_blueprint.has_attribute("noise_lat_bias"):
            gnss_blueprint.set_attribute("noise_lat_bias", "0.0")  # Latitude bias mean
        if gnss_blueprint.has_attribute("noise_lat_stddev"):
            gnss_blueprint.set_attribute(
                "noise_lat_stddev", "0.0"
            )  # Latitude standard deviation
        if gnss_blueprint.has_attribute("noise_lon_bias"):
            gnss_blueprint.set_attribute("noise_lon_bias", "0.0")  # Longitude bias mean
        if gnss_blueprint.has_attribute("noise_lon_stddev"):
            gnss_blueprint.set_attribute(
                "noise_lon_stddev", "0.0"
            )  # Longitude standard deviation
        if gnss_blueprint.has_attribute("noise_seed"):
            gnss_blueprint.set_attribute(
                "noise_seed", "0"
            )  # Pseudorandom seed for repeatable measurements
        if gnss_blueprint.has_attribute("sensor_tick"):
            gnss_blueprint.set_attribute(
                "sensor_tick", "0.0"
            )  # Sensor tick interval in simulation seconds

        # Spawn the sensor
        self.sensor = world.spawn_actor(
            blueprint=gnss_blueprint,
            transform=carla.Transform(carla.Location(x=1.0, y=0.0, z=2.8)),
            attach_to=self._parent,
            attachment=carla.AttachmentType.Rigid,
        )

        # We need to pass the lambda a weak reference to self to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda step: GNSSSensor._on_gnss_step(weak_self, step))
        self._logger.info("GNSS sensor online and listening every simulation step.")

    @staticmethod
    def _on_gnss_step(
        weak_self: weakref.ReferenceType, step: carla.GNSSMeasurement
    ) -> None:
        """
        Handle GNSS sensor data for a simulation step.

        Updates the GNSSSensor instance with the latest GNSS measurement.

        Args:
            weak_self (weakref.ReferenceType): Weak reference to the GNSSSensor instance.
            step (carla.GNSSMeasurement): Current GNSS measurement from the simulation.

        Returns:
            None
        """
        self = weak_self()
        if not self:
            # Exit if the GNSSSensor instance has been garbage collected
            return

        # Update instance attributes with current GNSS measurement
        self.frame = step.frame
        self.timestamp = step.timestamp
        self.lat = step.latitude
        self.lon = step.longitude
        self.alt = step.altitude


class IMUSensor(object):
    pass