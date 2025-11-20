"""_summary_
sensors.py

Sensor wrapper classes for CARLA simulation.

This module provides stateful wrappers for CARLA's asynchronous sensors, including collision, lane invasion, GNSS, and IMU sensors. Each class offers a clean, event-driven interface for attaching sensors to actors and accessing their latest data or event history in a structured form.
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


class CollisionSensor:
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
        self._parent = parent_actor
        self._logger = logging.getLogger(__name__)
        self.history = []

        world = self._parent.get_world()
        collision_blueprint = world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        # Spawn the sensor
        self.sensor = world.spawn_actor(
            blueprint=collision_blueprint,
            transform=carla.Transform(),
            attach_to=self._parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

        # Use a weak reference to prevent circular references between sensor and callback
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision_event_callback(weak_self, event)
        )
        self._logger.debug(
            f"{get_actor_display_name(self.sensor)} sensor online and listening for events."
        )

    def get_collision_history(
        self,
    ) -> List[Tuple[int, float, float, float, float, int, str]]:
        """
        Return a copy of the recorded collision history.

        Each entry contains:`(frame, timestamp, impulse_x, impulse_y, impulse_z, other_actor_id, other_actor_type_id)`.

        Returns:
            List[Tuple[]]: A shallow copy of the collision events history.
        """
        return self.history.copy()  # Return a shallow copy to prevent external mutation

    @staticmethod
    def _on_collision_event_callback(
        weak_self: weakref.ReferenceType, event: carla.CollisionEvent
    ) -> None:
        """
        Handles collision events from a CARLA collision sensor.

        This callback logs collision details and stores event data in a bounded history buffer.

        Args:
            weak_self (weakref.ReferenceType): Weak reference to the `CollisionSensor` instance.
            event (carla.CollisionEvent): Collision event data from CARLA.

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

        # self._logger.warning(
        #     f"Collided with {actor_type} (Impact Intensity: {collision_intensity:.2f} N·s)."
        # )

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


class GNSSSensor:
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
        self._gnss_tick = 1.0 / 10.0  # Sampling rate

        # Initialize sensor data attributes
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.frame = 0
        self.timestamp = 0.0

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
                "sensor_tick", str(self._gnss_tick)
            )  # Sensor tick interval in simulation seconds

        # Spawn the sensor
        self.sensor = world.spawn_actor(
            blueprint=gnss_blueprint,
            transform=carla.Transform(carla.Location(x=1.0, y=0.0, z=2.8)),
            attach_to=self._parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

        # We need to pass the lambda a weak reference to self to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda step: GNSSSensor._on_gnss_step_callback(weak_self, step)
        )
        self._logger.debug(
            f"{get_actor_display_name(self.sensor)} sensor online and listening at {1 / self._gnss_tick:.0f} Hz (noise-free)."
        )

    @staticmethod
    def _on_gnss_step_callback(
        weak_self: weakref.ReferenceType, step: carla.GnssMeasurement
    ) -> None:
        """
        Processes GNSS measurements from a CARLA GNSS sensor.

        This callback updates the GNSSSensor instance with the latest measurement values.

        Args:
            weak_self (weakref.ReferenceType): Weak reference to the `GNSSSensor` instance.
            step (carla.GnssMeasurement): GNSS measurement data containing frame, timestamp, and geolocation coordinates.

        Returns:
            None
        """
        self = weak_self()
        if not self:
            # Exit if the GNSSSensor instance has been garbage collected
            return

        # Update sensor data attributes with current GNSS measurement
        self.frame = step.frame
        self.timestamp = step.timestamp
        self.lat = step.latitude
        self.lon = step.longitude
        self.alt = step.altitude


class IMUSensor:
    """
    Stateful wrapper for an IMU sensor.
    """

    def __init__(self, parent_actor: carla.Actor) -> None:
        """Attach and configure an IMU sensor to a CARLA actor.

        The IMU sensor is spawned with all noise parameters disabled to ensure deterministic readings during simulation.

        Args:
            parent_actor (carla.Actor): The actor to which the IMU sensor is attached.

        Returns:
            None
        """
        self.sensor = None
        self._parent = parent_actor
        self._logger = logging.getLogger(__name__)
        self._imu_tick = 1.0 / 100.0  # Sampling rate

        # Initialize sensor data attributes
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        self.frame = 0
        self.timestamp = 0.0

        world = self._parent.get_world()
        imu_blueprint = world.get_blueprint_library().find("sensor.other.imu")

        # Configure IMU blueprint attributes for deterministic output
        if imu_blueprint.has_attribute("noise_accel_stddev_x"):
            imu_blueprint.set_attribute(
                "noise_accel_stddev_x", "0.0"
            )  # Acceleration standard deviation (X axis)
        if imu_blueprint.has_attribute("noise_accel_stddev_y"):
            imu_blueprint.set_attribute(
                "noise_accel_stddev_y", "0.0"
            )  # Acceleration standard deviation (Y axis)
        if imu_blueprint.has_attribute("noise_accel_stddev_z"):
            imu_blueprint.set_attribute(
                "noise_accel_stddev_z", "0.0"
            )  # Acceleration standard deviation (Z axis)

        if imu_blueprint.has_attribute("noise_gyro_bias_x"):
            imu_blueprint.set_attribute(
                "noise_gyro_bias_x", "0.0"
            )  # Gyroscope bias mean (X axis)
        if imu_blueprint.has_attribute("noise_gyro_bias_y"):
            imu_blueprint.set_attribute(
                "noise_gyro_bias_y", "0.0"
            )  # Gyroscope bias mean (Y axis)
        if imu_blueprint.has_attribute("noise_gyro_bias_z"):
            imu_blueprint.set_attribute(
                "noise_gyro_bias_z", "0.0"
            )  # Gyroscope bias mean (Z axis)

        if imu_blueprint.has_attribute("noise_gyro_stddev_x"):
            imu_blueprint.set_attribute(
                "noise_gyro_stddev_x", "0.0"
            )  # Gyroscope standard deviation (X axis)
        if imu_blueprint.has_attribute("noise_gyro_stddev_y"):
            imu_blueprint.set_attribute(
                "noise_gyro_stddev_y", "0.0"
            )  # Gyroscope standard deviation (Y axis)
        if imu_blueprint.has_attribute("noise_gyro_stddev_z"):
            imu_blueprint.set_attribute(
                "noise_gyro_stddev_z", "0.0"
            )  # Gyroscope standard deviation (Z axis)

        if imu_blueprint.has_attribute("noise_seed"):
            imu_blueprint.set_attribute(
                "noise_seed", "0"
            )  # Pseudorandom seed for repeatable measurements

        if imu_blueprint.has_attribute("sensor_tick"):
            imu_blueprint.set_attribute(
                "sensor_tick", str(self._imu_tick)
            )  # Sensor tick interval in simulation seconds

        # Spawn the sensor
        self.sensor = world.spawn_actor(
            blueprint=imu_blueprint,
            transform=carla.Transform(),
            attach_to=self._parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

        # We need to pass the lambda a weak reference to self to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda step: IMUSensor._on_imu_step_callback(weak_self, step)
        )
        self._logger.debug(
            f"{get_actor_display_name(self.sensor)} sensor online and listening at {1 / self._imu_tick:.0f} Hz (noise-free)."
        )

    @staticmethod
    def _on_imu_step_callback(
        weak_self: weakref.ReferenceType, step: carla.IMUMeasurement
    ) -> None:
        """
        Handle incoming IMU measurements from a CARLA IMU sensor.

        This callback updates the `IMUSensor` instance attributes with the latest accelerometer, gyroscope, and compass data, applying value clamping to maintain numerical stability.

        Args:
            weak_self (weakref.ReferenceType): Weak reference to the `IMUSensor` instance.
            step (carla.IMUMeasurement): Current IMU measurement containing frame, timestamp, and raw sensor readings.

        Returns:
            None
        """
        self = weak_self()
        if not self:
            # Exit if the IMUSensor instance has been garbage collected
            return

        # Update sensor data attributes with current IMU measurement
        self.frame = step.frame
        self.timestamp = step.timestamp

        # Clamp sensor readings to avoid extreme or invalid values
        measurement_limits = (-99.9, 99.9)

        # Clamp accelerometer readings
        self.accelerometer = (
            max(
                measurement_limits[0], min(measurement_limits[1], step.accelerometer.x)
            ),
            max(
                measurement_limits[0], min(measurement_limits[1], step.accelerometer.y)
            ),
            max(
                measurement_limits[0], min(measurement_limits[1], step.accelerometer.z)
            ),
        )

        # Convert gyroscope readings from radians to degrees and clamp
        self.gyroscope = (
            max(
                measurement_limits[0],
                min(measurement_limits[1], math.degrees(step.gyroscope.x)),
            ),
            max(
                measurement_limits[0],
                min(measurement_limits[1], math.degrees(step.gyroscope.y)),
            ),
            max(
                measurement_limits[0],
                min(measurement_limits[1], math.degrees(step.gyroscope.z)),
            ),
        )

        # Convert compass reading from radians to degrees
        self.compass = math.degrees(step.compass)


class LaneInvasionSensor:
    """
    Stateful wrapper around CARLA's lane invasion sensor to monitor and record lane invasion events for a CARLA vehicle.

    Attaches a lane invasion sensor to a parent actor to detect when a vehicle crosses lane markings, maintain a history of events, and provide structured access to logged data.
    """

    def __init__(self, parent_actor: carla.Actor) -> None:
        """
        Attach a lane invasion sensor to a CARLA vehicle actor.

        The sensor listens for lane invasion events and records them in the history buffer. It can only be attached to vehicle-type actors.

        Args:
            parent_actor (carla.Actor): The vehicle actor to attach the lane invasion sensor to.

        Returns:
            None
        """
        self.sensor = None

        # Ensure the sensor is only attached to vehicle actors
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self._logger = logging.getLogger(__name__)
            self.history = []

            world = self._parent.get_world()
            invasion_blueprint = world.get_blueprint_library().find(
                "sensor.other.lane_invasion"
            )
            # Spawn the sensor
            self.sensor = world.spawn_actor(
                blueprint=invasion_blueprint,
                transform=carla.Transform(),
                attach_to=self._parent,
                attachment_type=carla.AttachmentType.Rigid,
            )

            # Use a weak reference to prevent circular references between sensor and callback
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: LaneInvasionSensor._on_invasion_event_callback(
                    weak_self, event
                )
            )
            self._logger.debug(
                f"{get_actor_display_name(self.sensor)} sensor online and listening for events."
            )

    def get_invasion_history(self) -> List[Tuple[int, float, str, str, str, float]]:
        """
        Return a copy of the lane invasion event history.

        Each entry in the history is a tuple: `(frame, timestamp, lane_type, lane_color, lane_change, width)`.

        Returns:
            List[Tuple[]]: Shallow copy of the recorded lane invasion events.
        """
        return self.history.copy()  # Return a shallow copy to prevent external mutation

    @staticmethod
    def _on_invasion_event_callback(
        weak_self: weakref.ReferenceType, event: carla.LaneInvasionEvent
    ) -> None:
        """
        Handles lane invasion events from a CARLA lane invasion sensor.

        This callback updates the `LaneInvasionSensor` instance attributes with the lane invasion event details and stores relevant data in the history buffer of the sensor.

        Args:
            weak_self (weakref.ReferenceType): Weak reference to the `LaneInvasionSensor` instance.
            event (carla.LaneInvasionEvent): Lane invasion event containing frame, timestamp, and crossed lane marking data.

        Returns:
            None
        """
        self = weak_self()
        if not self:
            # Exit if the LaneInvasionSensor instance has been garbage collected
            return

        # Collect formatted descriptions of all crossed lane markings
        lane_descriptions = []
        for marking in event.crossed_lane_markings:
            lane_type = str(marking.type).split(".")[-1]
            lane_color = str(marking.color)

            lane_descriptions.append(f"'{lane_color} {lane_type}'")

        # self._logger.info(f"Crossed lines {' and '.join(lane_descriptions)}.")

        # Store the event sensor data attributes
        for marking in event.crossed_lane_markings:
            self.history.append(
                (
                    event.frame,
                    event.timestamp,
                    str(marking.type),
                    str(marking.color),
                    str(marking.lane_change),
                    marking.width,
                )
            )
