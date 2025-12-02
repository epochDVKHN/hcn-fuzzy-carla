import logging
import sys
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


class FLC_FreeFlow:
    def __init__(self) -> None:
        # -------Define antecedents and consequents--------
        self.speed_error = ctrl.Antecedent(
            universe=np.arange(-10, 10.1, 0.1),  # km/h
            label="speed_error",
        )
        self.acceleration = ctrl.Antecedent(
            universe=np.arange(-3, 3.1, 0.1),  # m/s^2
            label="acceleration",
        )
        self.control_intent = ctrl.Consequent(
            universe=np.arange(
                -1, 1.01, 0.01
            ),  # -1 is maximum brake, 1 is maximum throttle
            label="control_intent",
            defuzzify_method="centroid",
        )

        # --------Define antecedent membership functions--------
        # Negative (Too Fast): Lazy braking slope (Length 3)
        self.speed_error["Negative"] = fuzz.trapmf(
            x=self.speed_error.universe,
            abcd=[-10, -10, -3, 0],
        )
        # Zero (On Target): Asymmetric stability
        self.speed_error["Zero"] = fuzz.trimf(
            x=self.speed_error.universe,
            abc=[-3, 0, 2],
        )
        # Positive (Too Slow): Responsive acceleration slope (Length 2)
        self.speed_error["Positive"] = fuzz.trapmf(
            x=self.speed_error.universe,
            abcd=[0, 2, 10, 10],
        )

        # Decelerate: Only fires if stronger than natural drag/engine braking (-1.5)
        self.acceleration["Decelerate"] = fuzz.trapmf(
            x=self.acceleration.universe,
            abcd=[-3, -3, -1.5, 0],
        )
        # Stable: Covers the natural drag range (-1.5 to 1.0)
        self.acceleration["Stable"] = fuzz.trimf(
            x=self.acceleration.universe,
            abc=[-1.5, 0, 1],
        )
        # Accelerate: Sensitive to active torque (>1.0)
        self.acceleration["Accelerate"] = fuzz.trapmf(
            x=self.acceleration.universe,
            abcd=[0, 1, 3, 3],
        )

        # --------Define consequent membership functions--------
        # Brake Side (Negative Intent)
        self.control_intent["Strong_Brake"] = fuzz.trapmf(
            x=self.control_intent.universe,
            abcd=[-1, -1, -0.5, 0.15],  # Negative Big
        )
        self.control_intent["Light_Brake"] = fuzz.trimf(
            x=self.control_intent.universe,
            abc=[-0.5, 0.15, 0.45],  # Negative Medium
        )
        # Neutral Pivot
        self.control_intent["Maintain"] = fuzz.trimf(
            x=self.control_intent.universe,
            abc=[0.15, 0.45, 0.6],  # Zero
        )
        # Throttle Side (Positive Intent)
        self.control_intent["Light_Accelerate"] = fuzz.trimf(
            x=self.control_intent.universe,
            abc=[0.45, 0.6, 0.8],  # Positive Medium
        )
        self.control_intent["Strong_Accelerate"] = fuzz.trapmf(
            x=self.control_intent.universe,
            abcd=[0.6, 0.8, 1, 1],  # Positive Big
        )

        # --------Define core fuzzy rules--------
        # Group 1: speed too slow (positive error) -> need to accelerate
        # If slowing down further, push hard
        self._rule1 = ctrl.Rule(
            antecedent=(self.speed_error["Positive"] & self.acceleration["Decelerate"]),
            consequent=(self.control_intent["Strong_Accelerate"] % 1.0),
            label="rule_1",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Standard P-Control catch-up
        self._rule2 = ctrl.Rule(
            antecedent=(self.speed_error["Positive"] & self.acceleration["Stable"]),
            consequent=(self.control_intent["Strong_Accelerate"] % 1.0),
            label="rule_2",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # If already accelerating, damp the response to prevent overshoot
        self._rule3 = ctrl.Rule(
            antecedent=(self.speed_error["Positive"] & self.acceleration["Accelerate"]),
            consequent=(self.control_intent["Light_Accelerate"] % 1.0),
            label="rule_3",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # Group 2: speed on target (zero error) -> neutral
        # To fight drag to hold speed (power bias)
        self._rule4 = ctrl.Rule(
            antecedent=(self.speed_error["Zero"] & self.acceleration["Decelerate"]),
            consequent=(self.control_intent["Light_Accelerate"] % 1.0),
            label="rule_4",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Cruising power
        self._rule5 = ctrl.Rule(
            antecedent=(self.speed_error["Zero"] & self.acceleration["Stable"]),
            consequent=(self.control_intent["Maintain"] % 1.0),
            label="rule_5",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Gravity check when going downhill
        self._rule6 = ctrl.Rule(
            antecedent=(self.speed_error["Zero"] & self.acceleration["Accelerate"]),
            consequent=(self.control_intent["Light_Brake"] % 1.0),
            label="rule_6",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # Group 3: speed too fast (negative error) -> need to brake or coast
        # "Lazy Driver": If slowing down naturally, just coast (Maintain)
        self._rule7 = ctrl.Rule(
            antecedent=(self.speed_error["Negative"] & self.acceleration["Decelerate"]),
            # TODO: change to Strong_Brake
            consequent=(self.control_intent["Light_Brake"] % 1.0),
            label="rule_7",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Drag isn't enough, apply gentle brakes
        self._rule8 = ctrl.Rule(
            antecedent=(self.speed_error["Negative"] & self.acceleration["Stable"]),
            consequent=(self.control_intent["Light_Brake"] % 1.0),
            label="rule_8",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        # Panic: speeding and accelerating
        self._rule9 = ctrl.Rule(
            antecedent=(self.speed_error["Negative"] & self.acceleration["Accelerate"]),
            consequent=(self.control_intent["Strong_Brake"] % 1.0),
            label="rule_9",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # --------Assemble the fuzzy control system--------
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

    def compute_control_command(self, speed_error: float, acceleration: float) -> float:
        self.control_simulation.inputs(
            {
                # Clamp inputs to universe of discourse to prevent undefined behavior at the edges
                "speed_error": clamp_value(speed_error, -10, 10),
                "acceleration": clamp_value(acceleration, -3, 3),
            }
        )
        # Compute FLC
        self.control_simulation.compute()
        command_value = self.control_simulation.output["control_intent"]

        # Inspect command contribution from each term of the consequent membership functions
        command_contribution = self._inspect_command_contribution(command_value)

        # Map command value to CARLA throttle and brake controls
        if command_value >= 0:
            throttle = command_value
            brake = 0.0
        else:
            throttle = 0.0
            # Cap brake for comfort
            # brake = min(0.3, abs(command_value))
            brake = abs(command_value)

        throttle_brake_command = {"throttle": throttle, "brake": brake}

        return throttle_brake_command, command_contribution

    def _inspect_command_contribution(self, command_value: float):
        active_terms = {}

        # Iterate through all defined consequent terms
        for label in self.control_intent.terms:
            term = self.control_intent[label]
            # Determine degree of membership of the specific output value to this consequent membership term
            degree = fuzz.interp_membership(
                x=self.control_intent.universe,
                xmf=term.mf,
                xx=command_value,
                zero_outside_x=False,
            )
            if degree > 0.0:
                clean_degree = round(float(degree), 2)
                clean_label = label.replace("_", " ")
                active_terms[clean_label] = clean_degree

        return active_terms

    def visualize_system(self) -> None:
        # Visualize antecedent and consequent membership functions
        self.speed_error.view(sim=self.control_simulation)
        plt.show()
        self.acceleration.view(sim=self.control_simulation)
        plt.show()
        self.control_intent.view(sim=self.control_simulation)
        plt.show()

        # Visualize fuzzy rules
        # self._rule1.view()
        # plt.show()
        # self._rule2.view()
        # plt.show()
        # self._rule3.view()
        # plt.show()
        # self._rule4.view()
        # plt.show()
        # self._rule5.view()
        # plt.show()
        # self._rule6.view()
        # plt.show()
        # self._rule7.view()
        # plt.show()
        # self._rule8.view()
        # plt.show()
        # self._rule9.view()
        # plt.show()

        # Visualize fuzzy control system
        self.control_system.view()
        plt.show()
        self.control_system.view_n()
        plt.show()


class FLC_Bend:
    def __init__(self) -> None:
        pass


class FLC_Follow:
    def __init__(self) -> None:
        pass


class FLC_Emergency:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    flc = FLC_FreeFlow()
    # flc.visualize_system()
    flc.control_simulation.inputs({"speed_error": 0.0, "acceleration": 0.0})
    flc.control_simulation.compute()
    print(flc.control_simulation.output["control_intent"])
    flc.speed_error.view(sim=flc.control_simulation)
    plt.show()
    flc.acceleration.view(sim=flc.control_simulation)
    plt.show()
    flc.control_intent.view(sim=flc.control_simulation)
    plt.show()
    flc.compute_control_command(0.0, 0.0)
