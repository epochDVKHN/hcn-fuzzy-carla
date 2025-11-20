import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from skfuzzy import control as ctrl

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)

# Include the project root for relative module imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import clamp_value  # noqa: E402


class FuzzySteeringControl(object):
    def __init__(self) -> None:
        # Define antecedents and consequents
        self.lat_error = ctrl.Antecedent(
            universe=np.arange(-3.0, 3.1, 0.1),
            label="lat_error",
        )
        self.ang_error = ctrl.Antecedent(
            universe=np.arange(-5.0, 5.1, 0.1),
            label="ang_error",
        )
        self.steer_intent = ctrl.Consequent(
            universe=np.arange(-1.0, 1.01, 0.01),
            label="steer_intent",
            defuzzify_method="centroid",
        )

        # Define antecedent membership functions
        self.lat_error["ErrorRight"] = fuzz.trapmf(
            x=self.lat_error.universe, abcd=[-5.0, -5.0, -0.8, 0.0]
        )
        self.lat_error["ErrorCenter"] = fuzz.trimf(
            x=self.lat_error.universe, abc=[-0.8, 0.0, 0.8]
        )
        self.lat_error["ErrorLeft"] = fuzz.trapmf(
            x=self.lat_error.universe, abcd=[0.0, 0.8, 5.0, 5.0]
        )

        self.ang_error["AngleRight"] = fuzz.trapmf(
            x=self.ang_error.universe, abcd=[-5.0, -5.0, -2.0, 0.0]
        )
        self.ang_error["AngleCenter"] = fuzz.trimf(
            x=self.ang_error.universe, abc=[-2.0, 0.0, 2.0]
        )
        self.ang_error["AngleLeft"] = fuzz.trapmf(
            x=self.ang_error.universe, abcd=[0.0, 2.0, 5.0, 5.0]
        )

        # Define consequent membership functions
        self.steer_intent["SteerLeft"] = fuzz.trapmf(
            x=self.steer_intent.universe, abcd=[-1.0, -1.0, -0.2, -0.1]
        )
        self.steer_intent["NoSteer"] = fuzz.trimf(
            x=self.steer_intent.universe, abc=[-0.2, 0.0, 0.2]
        )
        self.steer_intent["SteerRight"] = fuzz.trapmf(
            x=self.steer_intent.universe, abcd=[0.1, 0.2, 1.0, 1.0]
        )

        # Define core fuzzy rules
        self._rule1 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorCenter"] & self.ang_error["AngleCenter"]),
            consequent=(self.steer_intent["NoSteer"] % 1.0),
            label="rule_1",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule2 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorLeft"] | self.ang_error["AngleLeft"]),
            consequent=(self.steer_intent["SteerRight"] % 1.0),
            label="rule_2",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule3 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorRight"] | self.ang_error["AngleRight"]),
            consequent=(self.steer_intent["SteerLeft"] % 1.0),
            label="rule_3",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Dampening rules to stabilize the car when it is correcting itself
        self._rule4 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorRight"] & self.ang_error["AngleLeft"]),
            consequent=(self.steer_intent["NoSteer"] % 1.0),
            label="rule_4",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        self._rule5 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorLeft"] & self.ang_error["AngleRight"]),
            consequent=(self.steer_intent["NoSteer"] % 1.0),
            label="rule_5",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # Assemble the fuzzy control system
        self.control_system = ctrl.ControlSystem(
            rules=[
                self._rule1,
                self._rule2,
                self._rule3,
                # self._rule4,
                # self._rule5,
            ],
        )
        self.control_simulation = ctrl.ControlSystemSimulation(
            control_system=self.control_system,
            clip_to_bounds=True,
            cache=True,
            flush_after_run=10,
            lenient=True,
        )

    def compute_steer_intent(
        self, lateral_error: float, angular_error: float, curve_history: deque
    ) -> float:
        self.control_simulation.inputs(
            {
                "lat_error": clamp_value(lateral_error, -3.0, 3.0),  # lateral_error
                "ang_error": clamp_value(angular_error, -5.0, 5.0),  # angular_error
            }
        )
        # is_curve_in_history = any(abs(angle_diff) > 0 for angle_diff in curve_history)
        # if not is_curve_in_history:
        #     self.steer_intent.defuzzify_method = "bisector"
        # else:
        #     self.steer_intent.defuzzify_method = "centroid"
        self.control_simulation.compute()
        return self.control_simulation.output["steer_intent"]

    def visualize(self):
        # Visualize antecedent and consequent membership functions
        self.lat_error.view()
        plt.show()
        self.ang_error.view()
        plt.show()
        self.steer_intent.view()
        plt.show()

        # Visualize fuzzy control system
        self.control_system.view()
        plt.show()
        self.control_system.view_n()
        plt.show()


if __name__ == "__main__":
    flc_low = FuzzySteeringControl()
    flc_low.visualize()
    flc_low.control_simulation.inputs({"lat_error": 8.43, "ang_error": -111.34})
    flc_low.control_simulation.compute()
    print(flc_low.control_simulation.output["steer_intent"])
    flc_low.steer_intent.view(sim=flc_low.control_simulation)
    plt.show()
