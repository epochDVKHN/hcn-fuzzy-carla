import logging

import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from skfuzzy import control as ctrl

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)


class LowSpeedControl(object):
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
        self.steer = ctrl.Consequent(
            universe=np.arange(-1.0, 1.01, 0.01),
            label="steer",
            defuzzify_method="centroid",
        )

        # Define antecedent membership functions
        self.lat_error["ErrorRight"] = fuzz.trapmf(
            x=self.lat_error.universe, abcd=[-60.0, -60.0, -0.8, 0.0]
        )
        self.lat_error["ErrorCenter"] = fuzz.trimf(
            x=self.lat_error.universe, abc=[-0.8, 0.0, 0.8]
        )
        self.lat_error["ErrorLeft"] = fuzz.trapmf(
            x=self.lat_error.universe, abcd=[0.0, 0.8, 60.0, 60.0]
        )

        self.ang_error["AngleRight"] = fuzz.trapmf(
            x=self.ang_error.universe, abcd=[-60.0, -60.0, -2.0, 0.0]
        )
        self.ang_error["AngleCenter"] = fuzz.trimf(
            x=self.ang_error.universe, abc=[-2.0, 0.0, 2.0]
        )
        self.ang_error["AngleLeft"] = fuzz.trapmf(
            x=self.ang_error.universe, abcd=[0.0, 2.0, 60.0, 60.0]
        )

        # Define consequent membership functions
        self.steer["SteerHardLeft"] = fuzz.trapmf(
            x=self.steer.universe, abcd=[-1.0, -1.0, -0.5, -0.3]
        )
        self.steer["SteerLeft"] = fuzz.trimf(
            x=self.steer.universe, abc=[-0.5, -0.3, 0.0]
        )
        self.steer["NoSteer"] = fuzz.trimf(x=self.steer.universe, abc=[-0.1, 0.0, 0.1])
        self.steer["SteerRight"] = fuzz.trimf(
            x=self.steer.universe, abc=[0.0, 0.3, 0.5]
        )
        self.steer["SteerHardRight"] = fuzz.trapmf(
            x=self.steer.universe, abcd=[0.3, 0.5, 1.0, 1.0]
        )

        # Define core fuzzy rules
        # Rule 1 (Perfect): Vehicle on path and pointing straight
        self._rule1 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorCenter"] & self.ang_error["AngleCenter"]),
            consequent=(self.steer["NoSteer"] % 1.0),
            label="rule_1",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Rules 2-3 (Correcting Angle): Vehicle on path but pointing wrong
        self._rule2 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorCenter"] & self.ang_error["AngleLeft"]),
            consequent=(self.steer["SteerRight"] % 1.0),
            label="rule_2",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule3 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorCenter"] & self.ang_error["AngleRight"]),
            consequent=(self.steer["SteerLeft"] % 1.0),
            label="rule_3",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Rules 4-5 (Standard Correction): Vehicle off path but pointing straight
        self._rule4 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorLeft"] & self.ang_error["AngleCenter"]),
            consequent=(self.steer["SteerRight"] % 1.0),
            label="rule_4",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule5 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorRight"] & self.ang_error["AngleCenter"]),
            consequent=(self.steer["SteerLeft"] % 1.0),
            label="rule_5",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Rules 6-7 (Critical Correction): Vehicle off path and pointing further away
        self._rule6 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorLeft"] & self.ang_error["AngleLeft"]),
            consequent=(self.steer["SteerHardRight"] % 1.0),
            label="rule_6",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule7 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorRight"] & self.ang_error["AngleRight"]),
            consequent=(self.steer["SteerHardLeft"] % 1.0),
            label="rule_7",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Rules 8-9 (Dampening): Vehicle off path but already correcting
        self._rule8 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorLeft"] & self.ang_error["AngleRight"]),
            consequent=(self.steer["NoSteer"] % 1.0),
            label="rule_8",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule9 = ctrl.Rule(
            antecedent=(self.lat_error["ErrorRight"] & self.ang_error["AngleLeft"]),
            consequent=(self.steer["NoSteer"] % 1.0),
            label="rule_9",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # Assemble the fuzzy control system
        self.control_system = ctrl.ControlSystem(
            rules=[
                self._rule1,
                self._rule2,
                self._rule3,
                self._rule4,
                self._rule5,
                self._rule6,
                self._rule7,
                self._rule8,
                self._rule9,
            ],
        )
        self.control_simulation = ctrl.ControlSystemSimulation(
            control_system=self.control_system,
            clip_to_bounds=True,
            cache=True,
            flush_after_run=10,
            lenient=True,
        )

    def compute_vehicle_control(
        self, lateral_error: float, angular_error: float
    ) -> float:
        try:
            self.control_simulation.inputs(
                {
                    "lat_error": lateral_error,
                    "ang_error": angular_error,
                }
            )
            self.control_simulation.compute()
            return self.control_simulation.output["steer"]
        except Exception as e:
            # Failsafe in case of compute error (e.g., if no rules fire)
            print(f"Fuzzy compute error: {e}")
            return 0.0

    def visualize(self):
        self.lat_error.view()
        plt.show()
        self.ang_error.view()
        plt.show()
        self.steer.view()
        plt.show()

        self.control_system.view()
        plt.show()
        self.control_system.view_n()
        plt.show()


if __name__ == "__main__":
    flc_low = LowSpeedControl()
    # flc_low.visualize()
    # # print(flc_low.compute_vehicle_control(1.5, 3.0))
    flc_low.control_simulation.inputs({"lat_error": 50.38, "ang_error": 50.61})
    flc_low.control_simulation.compute()
    print(flc_low.control_simulation.output["steer"])
    flc_low.steer.view(sim=flc_low.control_simulation)
    plt.show()
