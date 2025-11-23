"""pid_controller.py

PID Lateral Controller for CARLA

Provides a proportional-integral-derivative (PID) controller for lateral vehicle guidance in CARLA. The controller computes steering commands from the angular deviation between the vehicle's heading and a target waypoint. It supports gain tuning, optional lateral offsetting, steering saturation, and rate limiting for stable and realistic actuation dynamics.

Refactored from CARLA's Basic Agent vehicle controller implementation.
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
    """
    Implements a lateral PID controller for steering a vehicle toward target waypoints.

    The controller computes angular error between the vehicle heading and the waypoint direction, applies a proportional-integral-derivative correction, and enforces steering-rate and magnitude limits to maintain realistic behavior.
    """

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
    ) -> None:
        """
        Initialize the lateral PID controller.

        Args:
            vehicle (carla.Vehicle): Vehicle controlled by the planner.
            offset (float): Lateral deviation from the lane center.
            K_P (float): Proportional coefficient.
            K_I (float): Integral coefficient.
            K_D (float): Derivative coefficient.
            dt (float): Controller timestep.
            max_steer (float): Steering command saturation limit.
            rate_limit (float): Maximum allowed steering-rate change.
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._delta_time = dt
        self._offset = offset
        self._MAX_STEER = max_steer
        self._RATE_LIMIT = rate_limit
        print(
            f"K_P: {self._k_p}, K_I: {self._k_i}, K_D: {self._k_d}, dt: {self._delta_time}"
        )

        # Error buffer for integral and derivative terms
        self._error_buffer = deque(maxlen=10)

        # Track previous steering for smoothing logic
        self._prev_steer = self._vehicle.get_control().steer

    def compute_steer_control(
        self, target_waypoint: carla.Waypoint
    ) -> Dict[str, float]:
        """
        Compute the steering command toward a waypoint with rate limiting.

        Computes a raw steering value from the internal PID controller and applies rate limiting and clamping to enforce realistic steering dynamics.

        Args:
            target_waypoint (carla.Waypoint): Desired waypoint to track.

        Returns:
            steer_control(Dict[str, float]): A dictionary containing the key `"steer"` with the normalized steering output.
        """
        # Raw PID steering command based purely on geometric error
        current_raw_steer = self._calculate_pid_steer(
            target_waypoint, self._vehicle.get_transform()
        )

        # Rate limit: restrict steering delta change per timestep
        delta_steer = current_raw_steer - self._prev_steer
        rate_limit_steer = self._prev_steer + np.clip(
            delta_steer, -self._RATE_LIMIT, self._RATE_LIMIT
        )

        # Clamp to symmetric steering bounds
        steer_control = np.clip(rate_limit_steer, -self._MAX_STEER, self._MAX_STEER)

        # Update steering for the next control step
        self._prev_steer = steer_control

        return {"steer": steer_control}

    def _calculate_pid_steer(
        self, waypoint: carla.Waypoint, vehicle_transform: carla.Transform
    ) -> float:
        """
        Compute steering control using a PID controller to follow a waypoint.

        Calculates the steering angle required for the vehicle to align with a target waypoint, optionally applying a lateral offset. Uses proportional, integral, and derivative (PID) terms based on the angular error between the vehicle's heading and the waypoint direction.

        Args:
            waypoint (carla.Waypoint): Target waypoint to follow.
            vehicle_transform (carla.Transform): Current vehicle position and orientation.

        Returns:
            pid_steer (float): Steering control value in range [-1.0, 1.0].
        """
        # Extract vehicle position and forward direction
        veh_location = vehicle_transform.location
        veh_vector = vehicle_transform.get_forward_vector()
        veh_vector = np.array([veh_vector.x, veh_vector.y, 0.0])  # Flatten to 2D

        # Compute target waypoint position with optional lateral offset
        if self._offset != 0.0:
            wp_transform = waypoint.transform
            wp_right_vector = wp_transform.get_right_vector()
            wp_location = wp_transform.location + carla.Location(
                x=self._offset * wp_right_vector.x, y=self._offset * wp_right_vector.y
            )
        else:
            wp_location = waypoint.transform.location

        # Vector from vehicle to waypoint
        wp_vector = np.array(
            [wp_location.x - veh_location.x, wp_location.y - veh_location.y, 0.0]
        )

        # Precompute normalization factor for angular error
        wp_veh_norm = np.linalg.norm(wp_vector) * np.linalg.norm(veh_vector)

        # Compute angular error using cross and dot product
        _cross_prod = np.cross(veh_vector, wp_vector)
        if wp_veh_norm == 0.0:
            _dot_prod = 0.0
        else:
            _dot_prod = math.acos(
                np.clip(np.dot(wp_vector, veh_vector) / wp_veh_norm, -1.0, 1.0)
            )

        # Determine sign of angle based on cross product
        if _cross_prod[2] < 0.0:
            _dot_prod *= -1.0

        # Update error buffer for PID calculation
        self._error_buffer.append(_dot_prod)

        # Compute derivative and integral terms
        if len(self._error_buffer) >= 2:
            _deriv_error = (
                self._error_buffer[-1] - self._error_buffer[-2]
            ) / self._delta_time
            _integ_error = sum(self._error_buffer) * self._delta_time
        else:
            _deriv_error = 0.0
            _integ_error = 0.0

        # Compute and clamp PID output
        pid_steer = (
            (self._k_p * _dot_prod)
            + (self._k_i * _integ_error)
            + (self._k_d * _deriv_error)
        )

        return np.clip(pid_steer, -1.0, 1.0)

    def set_offset(self, offset: float) -> None:
        """
        Set the vehicle's lateral offset from the center line.

        Args:
            offset (float): Lateral displacement from the lane center.

                - Positive values shift the vehicle to the right.
                - Negative values shift the vehicle to the left.
                - Large magnitudes may cause unsafe lane behavior.
        """
        self._offset = offset

    def update_gains(self, K_P: float, K_I: float, K_D: float, dt: float) -> None:
        """
        Update the PID controller gains and time step.

        Args:
            K_P (float): Proportional gain.
            K_I (float): Integral gain.
            K_D (float): Derivative gain.
            dt (float): Time step in seconds.
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._delta_time = dt
