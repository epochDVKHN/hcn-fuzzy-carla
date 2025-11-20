import logging
import sys
from collections import deque
from pathlib import Path
from typing import Dict

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)

# Include the project root for relative module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.lateral_control.fuzzy_definitions import FuzzySteeringControl  # noqa: E402
from utils.helpers import clamp_value  # noqa: E402


class FuzzyLateralGeometricControl:
    def __init__(self):
        self._speed_sensitivity_factor = 0.5
        self._road_curve_history = deque(maxlen=10)
        self._curve_cooldown = 0
        self._COOLDOWN_FRAMES = 20
        self._lateral_controller = FuzzySteeringControl()

    def compute_steer_control(
        self,
        lateral_error: float,
        angular_error: float,
        angle_diff: float,
        speed: float,
    ) -> Dict[str, float]:
        self._road_curve_history.append(angle_diff)

        steer_intent = self._lateral_controller.compute_steer_intent(
            lateral_error, angular_error, self._road_curve_history
        )

        raw_steer = steer_intent + self._compute_steering_curvature_degree(angle_diff)

        steer_gain = self._compute_gain_scheduler(
            speed, self._road_curve_history, angle_diff
        )
        steer_control = clamp_value(raw_steer * steer_gain, -1.0, 1.0)

        return {"steer": steer_control}

    def _compute_gain_scheduler(
        self, current_speed: float, curve_history: deque, road_angle_diff: float
    ) -> float:
        # We check if ANY value in our recent history is significant.
        is_curve_in_history = any(abs(angle_diff) > 0 for angle_diff in curve_history)

        if is_curve_in_history > 0:
            # When driving at a bend or turn, the gain should be increased to allow for a larger steering output
            target_sensitivity = 0.05
            # self._speed_sensitivity_factor = 0.05
            # print(f"Straight | Factor: {self._speed_sensitivity_factor}")
            # Reset the time to full duration
            self._curve_cooldown = self._COOLDOWN_FRAMES
        elif self._curve_cooldown > 0:
            # Keep the curve settings to let the car settle
            target_sensitivity = 0.05
            self._speed_sensitivity_factor = 0.05
            self._curve_cooldown -= 1
        else:
            # When driving straight, the gain should be decreased to limit the steering output to reduce oscillations
            # Curve is gone and timer is finished
            target_sensitivity = 0.5
            self._speed_sensitivity_factor = 0.5
            # print(f"Turn | Factor: {self._speed_sensitivity_factor}")

        # Smooth the sensitivity transition using linear interpolation
        self._compute_sensitivity_interpolation(target_sensitivity)

        # Hyperbolic inverse function
        # Formula: 1 / (1 + v * k)
        gain = 1.0 / (1.0 + (current_speed * self._speed_sensitivity_factor))

        return gain

    def _compute_sensitivity_interpolation(self, target_sensitivity):
        alpha = 1.0
        self._speed_sensitivity_factor = (
            1.0 - alpha
        ) * self._speed_sensitivity_factor + (alpha * target_sensitivity)

    def _compute_steering_curvature_degree(self, curvature_degree: float) -> float:
        # Feedforward geometric controller that reacts to the curvature of the road
        curvature_gain = 0
        curvature_steer = curvature_degree * curvature_gain

        return curvature_steer
