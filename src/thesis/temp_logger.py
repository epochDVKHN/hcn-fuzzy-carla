"""
A dedicated data logger for thesis scenarios.

This module provides a DataLogger class that:
- Uses pathlib for modern path management.
- Is initialized with the full stack of measurement sensors.
- Logs stateful sensor data (IMU, GNSS) at each 'tick'.
- Collects event-based sensor data (Collision, Lane Invasion) internally.
- Saves all data to clean, separate CSV files upon completion.
- Creates empty CSVs with headers if no data was logged for a given category.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import carla
import numpy as np
import pandas as pd

# Use TYPE_CHECKING to avoid circular imports at runtime
# while still providing full type hints for the IDE.
if TYPE_CHECKING:
    from .sensors import CollisionSensor, GnssSensor, IMUSensor, LaneInvasionSensor

# Get a logger for this module
logger = logging.getLogger(__name__)


class DataLogger:
    """
    A stateful data logger that collects vehicle information
    at each step and saves it to separate, clean CSV files.
    """

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        imu_sensor: "IMUSensor",
        collision_sensor: "CollisionSensor",
        lane_invasion_sensor: "LaneInvasionSensor",
        gnss_sensor: "GnssSensor",
    ) -> None:
        """
        Initializes the logger and defines the data schemas.

        Args:
            vehicle (carla.Vehicle): The vehicle to monitor.
            imu_sensor (IMUSensor): The IMU sensor to monitor.
            collision_sensor (CollisionSensor): The collision sensor to monitor.
            lane_invasion_sensor (LaneInvasionSensor): The lane sensor to monitor.
            gnss_sensor (GnssSensor): The GNSS sensor to monitor.
        """
        self._vehicle = vehicle
        self._imu_sensor = imu_sensor
        self._collision_sensor = collision_sensor
        self._lane_invasion_sensor = lane_invasion_sensor
        self._gnss_sensor = gnss_sensor

        # --- Define schemas (column headers) ---
        self._time_series_columns = [
            "timestamp",
            "location_x",
            "location_y",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "speed_kmh",
            "accel_x",
            "accel_y",
            "accel_z",
            "imu_accel_x",
            "imu_accel_y",
            "imu_accel_z",
            "imu_gyro_x",
            "imu_gyro_y",
            "imu_gyro_z",
            "imu_compass",
            "gnss_lat",
            "gnss_lon",
            "gnss_alt",
            "control_throttle",
            "control_steer",
            "control_brake",
        ]
        self._collision_columns = [
            "frame",
            "timestamp",
            "impulse_x",
            "impulse_y",
            "impulse_z",
            "other_actor_id",
            "other_actor_type_id",
        ]
        self._lane_invasion_columns = [
            "frame",
            "timestamp",
            "type",
            "color",
            "lane_change",
            "width",
        ]

        # --- Data storage ---
        self.time_series_data = []
        logger.info("DataLogger initialized and ready to record.")

    def tick(self, world: "carla.World", control: "carla.VehicleControl") -> None:
        """
        Logs the current state of the vehicle and stateful sensors.
        This should be called at each simulation step (tick).
        """
        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds

        velocity = self._vehicle.get_velocity()
        acceleration = self._vehicle.get_acceleration()
        transform = self._vehicle.get_transform()

        frame_data = {
            "timestamp": timestamp,
            "location_x": transform.location.x,
            "location_y": transform.location.y,
            "velocity_x": velocity.x,
            "velocity_y": velocity.y,
            "velocity_z": velocity.z,
            "speed_kmh": 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            "accel_x": acceleration.x,
            "accel_y": acceleration.y,
            "accel_z": acceleration.z,
            "imu_accel_x": self._imu_sensor.accelerometer[0],
            "imu_accel_y": self._imu_sensor.accelerometer[1],
            "imu_accel_z": self._imu_sensor.accelerometer[2],
            "imu_gyro_x": self._imu_sensor.gyroscope[0],
            "imu_gyro_y": self._imu_sensor.gyroscope[1],
            "imu_gyro_z": self._imu_sensor.gyroscope[2],
            "imu_compass": self._imu_sensor.compass,
            "gnss_lat": self._gnss_sensor.lat,
            "gnss_lon": self._gnss_sensor.lon,
            "gnss_alt": self._gnss_sensor.alt,
            "control_throttle": control.throttle,
            "control_steer": control.steer,
            "control_brake": control.brake,
        }
        self.time_series_data.append(frame_data)

    def save(
        self, output_dir: Path = Path("results"), file_prefix: str = "log"
    ) -> None:
        """
        Saves all collected data to CSV files in the specified directory.
        If no data was logged, empty files with headers are created.

        Args:
            output_dir (pathlib.Path or str): The directory to save files in.
            file_prefix (str): The prefix for the filenames.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Save the Time-Series Data ---
        ts_path = output_dir / f"{file_prefix}_timeseries.csv"
        try:
            if not self.time_series_data:
                df_ts = pd.DataFrame(columns=self._time_series_columns)
                logger.info(
                    "Time-series data log is empty. Saving empty log with headers."
                )
            else:
                df_ts = pd.DataFrame(self.time_series_data)
                logger.info(f"Saving {len(df_ts)} time-series records to log.")

            df_ts.to_csv(ts_path, index=False)
            logger.info(f"Time-series data log successfully saved to {ts_path}")
        except Exception as e:
            logger.error(f"Error saving time-series data: {e}", exc_info=True)

        # --- 2. Save the Collision Sensor Data ---
        col_path = output_dir / f"{file_prefix}_collisions.csv"
        collision_history = self._collision_sensor.get_collision_history()

        try:
            if not collision_history:
                df_col = pd.DataFrame(columns=self._collision_columns)
                logger.info(
                    "No collision events recorded. Saving empty collision log with headers."
                )
            else:
                df_col = pd.DataFrame(
                    collision_history, columns=self._collision_columns
                )
                logger.info(f"Saving {len(df_col)} collision events to log.")

            df_col.to_csv(col_path, index=False)
            logger.info(f"Collision data log successfully saved to {col_path}")
        except Exception as e:
            logger.error(f"Error saving collision data: {e}", exc_info=True)

        # --- 3. Save the Lane Invasion Sensor Data ---
        lane_path = output_dir / f"{file_prefix}_lane_invasions.csv"
        invasion_history = self._lane_invasion_sensor.get_invasion_history()

        try:
            if not invasion_history:
                df_lane = pd.DataFrame(columns=self._lane_invasion_columns)
                logger.info(
                    "No lane invasion events recorded. Saving empty invasion log with headers."
                )
            else:
                df_lane = pd.DataFrame(
                    invasion_history, columns=self._lane_invasion_columns
                )
                logger.info(f"Saving {len(df_lane)} lane invasion events to log.")

            df_lane.to_csv(lane_path, index=False)
            logger.info(f"Lane invasion data log successfully saved to {lane_path}")
        except Exception as e:
            logger.error(f"Error saving lane invasion data: {e}", exc_info=True)
