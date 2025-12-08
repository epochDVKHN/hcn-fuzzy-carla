import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import carla
import pandas as pd

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)

# Add project root path for module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if TYPE_CHECKING:
    from utils.sensors import CollisionSensor, GNSSSensor, IMUSensor, LaneInvasionSensor


class DataLogger:
    def __init__(
        self,
        vehicle: carla.Actor,
        imu_sensor: "IMUSensor",
        collision_sensor: "CollisionSensor",
        lane_invasion_sensor: "LaneInvasionSensor",
        gnss_sensor: "GNSSSensor",
    ) -> None:
        self._vehicle = vehicle
        self._imu_sensor = imu_sensor
        self._collision_sensor = collision_sensor
        self._lane_invasion_sensor = lane_invasion_sensor
        self._gnss_sensor = gnss_sensor

        self._time_series_columns = [
            "timestamp",
            "frame",
            "location_x",
            "location_y",
            "forward_x",
            "forward_y",
            "forward_z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "target_speed_kmh",
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "imu_accelerometer_x",
            "imu_accelerometer_y",
            "imu_accelerometer_z",
            "imu_gyroscope_x",
            "imu_gyroscope_y",
            "imu_gyroscope_z",
            "imu_compass",
            "gnss_latitude",
            "gnss_longitude",
            "gnss_altitude",
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

        self.time_series_data: List[Dict[str, Any]] = []
        LOGGER.debug("DataLogger initialized.")

    def tick(self, world: carla.World, control: carla.VehicleControl) -> None:
        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame

        velocity = self._vehicle.get_velocity()
        acceleration = self._vehicle.get_acceleration()
        transform = self._vehicle.get_transform()
        forward_vector = transform.get_forward_vector()

        target_speed_kmh = self._vehicle.get_speed_limit()

        frame_data = {
            "timestamp": timestamp,
            "frame": frame,
            "location_x": transform.location.x,
            "location_y": transform.location.y,
            "forward_x": forward_vector.x,
            "forward_y": forward_vector.y,
            "forward_z": forward_vector.z,
            "velocity_x": velocity.x,
            "velocity_y": velocity.y,
            "velocity_z": velocity.z,
            "target_speed_kmh": target_speed_kmh,
            "acceleration_x": acceleration.x,
            "acceleration_y": acceleration.y,
            "acceleration_z": acceleration.z,
            "imu_accelerometer_x": self._imu_sensor.accelerometer[0],
            "imu_accelerometer_y": self._imu_sensor.accelerometer[1],
            "imu_accelerometer_z": self._imu_sensor.accelerometer[2],
            "imu_gyroscope_x": self._imu_sensor.gyroscope[0],
            "imu_gyroscope_y": self._imu_sensor.gyroscope[1],
            "imu_gyroscope_z": self._imu_sensor.gyroscope[2],
            "imu_compass": self._imu_sensor.compass,
            "gnss_latitude": self._gnss_sensor.lat,
            "gnss_longitude": self._gnss_sensor.lon,
            "gnss_altitude": self._gnss_sensor.alt,
            "control_throttle": control.throttle,
            "control_brake": control.brake,
            "control_steer": control.steer,
        }
        self.time_series_data.append(frame_data)

    def save(self, output_dir: Path, file_prefix: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Simulation results saved to [{output_dir.resolve()}].")

        # Time Series
        ts_path = output_dir / f"{file_prefix}_timeseries.csv"
        pd.DataFrame(
            data=self.time_series_data,
        ).to_csv(
            path_or_buf=ts_path,
            index=False,
            mode="w",
        )

        # Collisions
        col_path = output_dir / f"{file_prefix}_collisions.csv"
        col_history = self._collision_sensor.get_collision_history()
        pd.DataFrame(
            data=col_history,
            columns=self._collision_columns,
        ).to_csv(
            path_or_buf=col_path,
            index=False,
            mode="w",
        )

        # Lane Invasions
        lane_path = output_dir / f"{file_prefix}_lane_invasions.csv"
        inv_history = self._lane_invasion_sensor.get_invasion_history()
        pd.DataFrame(
            data=inv_history,
            columns=self._lane_invasion_columns,
        ).to_csv(
            path_or_buf=lane_path,
            index=False,
            mode="w",
        )
