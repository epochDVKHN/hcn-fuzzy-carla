import logging
import sys
from pathlib import Path

import carla
from agents.navigation.basic_agent import BasicAgent

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)

# Include the project root for relative module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.fuzzy_lateral.definitions import LowSpeedControl  # noqa: E402
from utils.helpers import compute_lateral_angular_errors  # noqa: E402


class FuzzyControlAgent(BasicAgent):
    def __init__(
        self,
        vehicle: carla.Vehicle,
        target_speed: float = 20,
        opt_dict: dict = {},
        debug: bool = False,
    ) -> None:
        super().__init__(vehicle, target_speed, opt_dict, debug)
        self._vehicle = vehicle
        self._steer_control = LowSpeedControl()
        self._debug = debug

    def run_step(self, frame_ID: int) -> carla.VehicleControl:
        pid_control = self._local_planner.run_step(self._debug)

        # Calculate lateral offset and heading error for the vehicle
        lateral_error, angular_error = compute_lateral_angular_errors(self)

        fuzzy_steer = self._steer_control.compute_vehicle_control(
            lateral_error=lateral_error,
            angular_error=angular_error,
        )

        if frame_ID % 10 == 0:
            print(
                f"Lateral error: {lateral_error:.2f} | Angular error: {angular_error:.2f} | Steer: {fuzzy_steer:.2f}"
            )

        fuzzy_control = carla.VehicleControl(
            throttle=pid_control.throttle,
            brake=pid_control.brake,
            steer=pid_control.steer,
            # steer=fuzzy_steer,
            hand_brake=pid_control.hand_brake,
            manual_gear_shift=pid_control.manual_gear_shift,
            gear=pid_control.gear,
        )

        return fuzzy_control
