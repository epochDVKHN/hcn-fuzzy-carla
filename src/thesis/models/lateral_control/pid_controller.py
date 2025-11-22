"""pid_controller.py

This module implements a basic PID controller for lateral control of a vehicle in CARLA simulation.
Refactored from the original code: https://github.com/carla-simulator/carla/blob/0.9.15/PythonAPI/carla/agents/navigation/controller.py
"""

import logging
import math
from collections import deque
from typing import Dict

import carla
import numpy as np

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)


class PIDLateralControl:
    def __init__(
        self,
        vehicle: carla.Vehicle,
        offset: float = 0.0,
        K_P: float = 1.95,
        K_I: float = 0.05,
        K_D: float = 0.2,
        dt: float = 1.0 / 20.0,
        max_steer: float = 0.8,
        rate_limit: float = 0.1,
    ):
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._delta_time = dt
        self._offset = offset
        self._MAX_STEER = max_steer
        self._RATE_LIMIT = rate_limit

        # Error buffer for integral and derivative terms
        self._error_buffer = deque(maxlen=10)

        # Track previous steering for smoothing logic
        self._prev_steer = self._vehicle.get_control().steer

    def compute_steer_control(
        self, target_waypoint: carla.Waypoint
    ) -> Dict[str, float]:
        # Calculate raw PID steer based on internal geometry math
        current_raw_steer = self._calculate_pid_steer(
            target_waypoint, self._vehicle.get_transform()
        )

        # Apply smoothing logic (rate limiting per step to mimic physical steering constraints)
        if current_raw_steer > self._prev_steer + self._RATE_LIMIT:
            current_raw_steer = self._prev_steer + self._RATE_LIMIT
        elif current_raw_steer < self._prev_steer - self._RATE_LIMIT:
            current_raw_steer = self._prev_steer - self._RATE_LIMIT

        # Clamp to maximum steering limits
        if current_raw_steer >= 0:
            steer_control = min(current_raw_steer, self._MAX_STEER)
        else:
            steer_control = max(current_raw_steer, -self._MAX_STEER)

        # Update previous steering for next step
        self._prev_steer = steer_control

        return {"steer": steer_control}

    def _calculate_pid_steer(
        self, waypoint: carla.Waypoint, vehicle_transform: carla.Transform
    ) -> float:
        # Get vehicle location and forward vector
        veh_location = vehicle_transform.location
        veh_vector = vehicle_transform.get_forward_vector()
        veh_vector = np.array([veh_vector.x, veh_vector.y, 0.0])

        # Compute target waypoint location, applying lateral offset if present
        if self._offset != 0.0:
            wp_transform = waypoint.transform
            wp_right_vector = wp_transform.get_right_vector()
            wp_location = wp_transform.location + carla.Location(
                x=self._offset * wp_right_vector.x, y=self._offset * wp_right_vector.y
            )
        else:
            wp_location = waypoint.transform.location

        # Compute vector from vehicle to waypoint
        wp_vector = np.array(
            [wp_location.x - veh_location.x, wp_location.y - veh_location.y, 0.0]
        )

        # Compute normalization factors for error computation
        wp_veh_norm = np.linalg.norm(wp_vector) * np.linalg.norm(veh_vector)

        # Compute cross product
        _cross_prod = np.cross(veh_vector, wp_vector)

        # Compute dot product
        if wp_veh_norm == 0.0:
            _dot_prod = 0.0
        else:
            _dot_prod = math.acos(
                np.clip(np.dot(wp_vector, veh_vector) / wp_veh_norm, -1.0, 1.0)
            )
        if _cross_prod[2] < 0.0:
            _dot_prod *= -1.0

        # Compute PID equation
        self._error_buffer.append(_dot_prod)
        if len(self._error_buffer) >= 2:
            _deriv_error = (
                self._error_buffer[-1] - self._error_buffer[-2]
            ) / self._delta_time
            _integ_error = sum(self._error_buffer) * self._delta_time
        else:
            _deriv_error = 0.0
            _integ_error = 0.0

        return np.clip(
            (self._k_p * _dot_prod)
            + (self._k_i * _integ_error)
            + (self._k_d * _deriv_error),
            -1.0,
            1.0,
        )

    def set_offset(self, offset: float) -> None:
        self._offset = offset

    def update_gains(self, K_P: float, K_I: float, K_D: float, dt: float) -> None:
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._delta_time = dt
