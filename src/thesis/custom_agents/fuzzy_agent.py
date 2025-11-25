import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import get_speed

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)

# Include the project root for relative module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.lateral_control.pid_controller import PIDLateralControl  # noqa: E402
from utils.helpers import compute_steering_deviations  # noqa: E402


class FuzzyControlAgent(BasicAgent):
    # Local planner look-ahead parameters sets (base_min_distance, distance_ratio) for lateral motion
    _STABILITY_MODE = (3.0, 0.5)  # For stable high-speed driving and lane changes
    _PRECISION_MODE = (1.2, 0.2)  # For precise low-speed driving especially on curves

    def __init__(
        self,
        vehicle: carla.Vehicle,
        target_speed: float = 20,
        opt_dict: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        if opt_dict is None:
            opt_dict = {}

        super().__init__(vehicle, target_speed, opt_dict, debug)

        self._vehicle = vehicle
        self._debug = debug

        # Initialise longitudinal controller
        self._longitudinal_controller = None

        # Initialise lateral controller
        _lateral_kwargs = opt_dict.get("lateral_control_dict", {})
        if "dt" not in _lateral_kwargs and "dt" in opt_dict:
            _lateral_kwargs["dt"] = opt_dict["dt"]

        self._lateral_controller = PIDLateralControl(
            vehicle=self._vehicle,
            offset=opt_dict.get("offset", 0.0),
            max_steer=opt_dict.get("max_steering", 0.8),
            rate_limit=opt_dict.get("rate_limit_steering", 0.1),
            **_lateral_kwargs,
        )

        # Initialise vehicle control
        self._control_kwargs = {
            "throttle": 0.0,
            "brake": 0.0,
            "steer": 0.0,
            "hand_brake": False,
            "reverse": False,
            "manual_gear_shift": False,
            "gear": 0,
        }

        self._road_curvature_history = deque(maxlen=35)

    def run_step(self, frame_ID: int) -> carla.VehicleControl:
        # Gather vehicle state data
        current_speed_kmh = get_speed(self._vehicle)
        current_offset = self._lateral_controller._offset
        current_angle_diff = self._get_road_curvature(lookahead_steps=25)
        self._road_curvature_history.append(current_angle_diff)

        # Adjust the local planner's "hit box" size based on speed and offset
        self._update_planner_lateral_lookahead(current_speed_kmh, abs(current_offset))
        # Update the local planner's waypoint navigation queue
        pid_control = self._local_planner.run_step(self._debug)

        # Compute steering deviations of the vehicle
        lateral_error, angular_error = compute_steering_deviations(self)

        # Compute steering control
        steer_control = self._lateral_controller.compute_steer_control(
            self._local_planner.target_waypoint
        )
        self._control_kwargs.update(steer_control)

        # if frame_ID % 5 == 0:
        #     print(
        #         f"Angle diff: {current_angle_diff} | Steer: {self._control_kwargs['steer']:.2f}"
        #     )

        # Update vehicle control
        agent_control = carla.VehicleControl(**self._control_kwargs)

        agent_control.throttle = pid_control.throttle
        agent_control.brake = pid_control.brake

        return agent_control

    def update_path_offset(self, new_offset: float) -> None:
        """
        Set the vehicle's lateral path offset.

        Args:
            new_offset (float): Lateral deviation from the road centerline (meters).
        """
        self._lateral_controller.set_offset(new_offset)
        self._local_planner.set_offset(new_offset)
        self._local_planner._offset = new_offset

    def _check_road_curvature(self) -> bool:
        """
        Determine whether the vehicle is in, approaching, or has just exited a curved road.

        Returns:
            in_curve_state (bool): True if any recent road curvature value indicates a non-straight segment.
        """
        in_curve_state = any(abs(angle) > 0 for angle in self._road_curvature_history)

        return in_curve_state

    def _get_road_curvature(self, lookahead_steps: int = 5) -> int:
        """
        Compute the signed angle difference (curvature) between the current waypoint and a future waypoint ahead.

        The corresponding signed angle represents the curvature of the road:
        - A positive value indicates a turn to the right.
        - A negative value indicates a turn to the left.
        - A value of zero indicates a straight road.

        Args:
            lookahead_steps (int): Number of waypoints ahead to consider.

        Returns:
            angle_diff (int): Signed angle difference in degrees, in the range [-180, 180], rounded to the nearest degree.
        """
        # Retrieve the future waypoint at the specified lookahead distance
        future_node = self._local_planner.get_incoming_waypoint_and_direction(
            steps=lookahead_steps
        )

        # If no future waypoint exists, assume straight road
        if future_node is None or future_node[0] is None:
            return 0.0
        future_waypoint = future_node[0]
        # future_road_option = future_node[1]

        # Get current waypoint from the local planner
        current_waypoint = self._local_planner.target_waypoint
        # current_road_option = self._local_planner.target_road_option

        # Extract yaw angles in degrees
        current_yaw = current_waypoint.transform.rotation.yaw
        future_yaw = future_waypoint.transform.rotation.yaw

        # Compute signed shortest rotation between current and future yaw
        angle_diff = (future_yaw - current_yaw + 180) % 360 - 180

        return round(angle_diff)

    def _update_planner_lateral_lookahead(
        self, current_speed_kmh: float, current_offset_abs: float
    ) -> None:
        """
        Selects lateral look-ahead parameters based on vehicle speed, lateral offset, and upcoming curvature.

        The method operates in two distinct modes:

        - *Precision Mode*:
        This mode applies a short look-ahead horizon to force the planner to follow the immediate tangent of the path.
        It is intended to maintain geometric accuracy in curves and at low speeds, reducing chordal errors that can otherwise lead to corner-cutting or curb encroachment.

        - *Stability Mode*:
        This mode applies a longer look-ahead horizon to dampen steering aggressiveness.
        It is used only on straight segments and under high-speed or large-offset conditions, which helps prevent steering oscillations and improves lateral stability during fast cruising or significant lateral corrections.

        Args:
            current_speed_kmh (float): Current vehicle speed in km/h.
            current_offset_abs (float): Absolute lateral deviation from the reference path in meters.
        """
        CURVE_CHECK = self._check_road_curvature()
        STABILITY_CHECK = current_speed_kmh > 40.0 or current_offset_abs > 0.5

        #  Switch to stability mode only on straight segments and only when conditions justify it
        if STABILITY_CHECK and not CURVE_CHECK:
            lateral_lookahead_params = self._STABILITY_MODE
        else:
            # Otherwise, default to precision mode for safer geometric tracking
            lateral_lookahead_params = self._PRECISION_MODE

        # Apply selected lookahead parameters to the local planner
        base_min_distance, distance_ratio = lateral_lookahead_params
        self._local_planner._base_min_distance = base_min_distance
        self._local_planner._distance_ratio = distance_ratio
