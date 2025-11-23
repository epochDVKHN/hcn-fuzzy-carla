import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import carla
from agents.navigation.basic_agent import BasicAgent

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

        self._lateral_error = 0.0
        self._angular_error = 0.0

    def run_step(self, frame_ID: int) -> carla.VehicleControl:
        pid_control = self._local_planner.run_step(self._debug)

        # Get road curvature, i.e. whether the vehicle is on a curved road or driving on a straight road
        angle_diff = self._get_road_curvature()

        # Compute steering control
        steer_control = self._lateral_controller.compute_steer_control(
            self._local_planner.target_waypoint
        )
        self._control_kwargs.update(steer_control)

        # Compute steering deviations of the vehicle
        self._lateral_error, self._angular_error = compute_steering_deviations(self)

        if frame_ID % 5 == 0:
            print(
                f"Lateral error: {self._lateral_error:.2f} | Angular error: {self._angular_error:.2f} | Angle diff: {angle_diff} | Steer: {self._control_kwargs['steer']:.2f}"
            )

        # Update vehicle control
        agent_control = carla.VehicleControl(**self._control_kwargs)

        agent_control.throttle = pid_control.throttle
        agent_control.brake = pid_control.brake
        # agent_control.steer = pid_control.steer

        return agent_control

    def update_path_offset(self, new_offset: float) -> None:
        self._lateral_controller.set_offset(new_offset)
        self._local_planner.set_offset(new_offset)
        self._local_planner._offset = new_offset

    def _get_road_curvature(self, lookahead_steps: int = 3) -> float:
        # Get the future waypoints
        future_node = self._local_planner.get_incoming_waypoint_and_direction(
            steps=lookahead_steps
        )
        # Assume straight if no road ahead
        if future_node is None or future_node[0] is None:
            return 0.0
        else:
            future_waypoint = future_node[0]
            # future_road_option = future_node[1]

        # Get current waypoint from the local planner
        current_waypoint = self._local_planner.target_waypoint
        # current_road_option = self._local_planner.target_road_option

        # Get waypoint yaws
        current_yaw = round(current_waypoint.transform.rotation.yaw, 0)
        future_yaw = round(future_waypoint.transform.rotation.yaw, 0)

        # Calculate signed angle difference ("shortest angle")
        angle_diff = (future_yaw - current_yaw + 180) % 360 - 180

        # if angle_diff > 0:
        #     print(
        #         f"Right turn: {angle_diff}, Current yaw: {current_yaw}, Future yaw: {future_yaw}"
        #     )
        # elif angle_diff < 0:
        #     print(
        #         f"Left turn: {angle_diff}, Current yaw: {current_yaw}, Future yaw: {future_yaw}"
        #     )
        # else:
        #     print(
        #         f"Straight: {angle_diff}, Current yaw: {current_yaw}, Future yaw: {future_yaw}"
        #     )

        return angle_diff
