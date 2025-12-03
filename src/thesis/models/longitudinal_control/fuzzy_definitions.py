import logging
import sys
from pathlib import Path
from typing import Dict

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
        """
        Initialize the fuzzy controller for longitudinal vehicle control.

        This sets up the fuzzy antecedents (speed error, acceleration), the control intent consequent, their membership functions, and the complete rule base governing throttle and brake outputs.
        """
        # -------------- Antecedents & Consequent --------------
        # Speed error: negative -> too fast, positive -> too slow (km/h)
        self.speed_error = ctrl.Antecedent(
            universe=np.arange(-10, 10.1, 0.1),
            label="speed_error",
        )
        # Acceleration: negative -> deceleration, positive -> active torque (m/s^2)
        self.acceleration = ctrl.Antecedent(
            universe=np.arange(-3, 3.1, 0.1),
            label="acceleration",
        )
        # Control intent: -1 = max brake, +1 = max throttle
        self.control_intent = ctrl.Consequent(
            universe=np.arange(-1, 1.01, 0.01),
            label="control_intent",
            defuzzify_method="centroid",
        )

        # ----------- Speed Error Memberships -----------
        self.speed_error["Negative"] = fuzz.trapmf(
            x=self.speed_error.universe,
            abcd=[-10, -10, -3, 0],
            # abcd=[-10, -10, -2, 0],
        )
        self.speed_error["Zero"] = fuzz.trimf(
            x=self.speed_error.universe,
            # abc=[-3, 0, 2],
            abc=[-2, 0, 2],
        )
        self.speed_error["Positive"] = fuzz.trapmf(
            x=self.speed_error.universe,
            abcd=[0, 2, 10, 10],
        )

        # ------------- Acceleration Memberships -------------
        self.acceleration["Decelerate"] = fuzz.trapmf(
            x=self.acceleration.universe,
            abcd=[-3, -3, -1.5, 0],
        )
        self.acceleration["Stable"] = fuzz.trimf(
            x=self.acceleration.universe,
            abc=[-1.5, 0, 1],
        )
        self.acceleration["Accelerate"] = fuzz.trapmf(
            x=self.acceleration.universe,
            abcd=[0, 1, 3, 3],
        )

        # ------------- Control Intent Memberships -------------
        # Strong braking region (Negative Big)
        self.control_intent["Strong_Brake"] = fuzz.trapmf(
            x=self.control_intent.universe,
            abcd=[-1, -1, -0.5, 0.15],
        )
        # Light brake around mild deceleration (Negative Medium)
        self.control_intent["Light_Brake"] = fuzz.trimf(
            x=self.control_intent.universe,
            abc=[-0.5, 0.15, 0.2],
        )
        # Neutral or hold-speed region (Zero)
        self.control_intent["Maintain"] = fuzz.trapmf(
            x=self.control_intent.universe,
            abcd=[0.15, 0.2, 0.5, 0.6],  # Zero
        )
        # Light throttle application (Positive Medium)
        self.control_intent["Light_Accelerate"] = fuzz.trimf(
            x=self.control_intent.universe,
            abc=[0.5, 0.65, 0.8],
        )
        # Strong throttle demand (Positive Big)
        self.control_intent["Strong_Accelerate"] = fuzz.trapmf(
            x=self.control_intent.universe,
            abcd=[0.65, 0.8, 1, 1],
        )

        # --------------- Rule Base ---------------
        # Too slow -> prioritize acceleration
        self._rule1 = ctrl.Rule(
            antecedent=(self.speed_error["Positive"] & self.acceleration["Decelerate"]),
            consequent=(self.control_intent["Strong_Accelerate"] % 1.0),
            label="rule_1",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule2 = ctrl.Rule(
            antecedent=(self.speed_error["Positive"] & self.acceleration["Stable"]),
            consequent=(self.control_intent["Strong_Accelerate"] % 1.0),
            label="rule_2",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule3 = ctrl.Rule(
            antecedent=(self.speed_error["Positive"] & self.acceleration["Accelerate"]),
            consequent=(self.control_intent["Light_Accelerate"] % 1.0),
            label="rule_3",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # On-target speed -> maintain or counter drift
        self._rule4 = ctrl.Rule(
            antecedent=(self.speed_error["Zero"] & self.acceleration["Decelerate"]),
            consequent=(self.control_intent["Light_Accelerate"] % 1.0),
            label="rule_4",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule5 = ctrl.Rule(
            antecedent=(self.speed_error["Zero"] & self.acceleration["Stable"]),
            consequent=(self.control_intent["Maintain"] % 1.0),
            label="rule_5",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule6 = ctrl.Rule(
            antecedent=(self.speed_error["Zero"] & self.acceleration["Accelerate"]),
            consequent=(self.control_intent["Light_Brake"] % 1.0),
            label="rule_6",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # Too fast -> braking or coasting
        self._rule7 = ctrl.Rule(
            antecedent=(self.speed_error["Negative"] & self.acceleration["Decelerate"]),
            consequent=(self.control_intent["Strong_Brake"] % 1.0),
            label="rule_7",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule8 = ctrl.Rule(
            antecedent=(self.speed_error["Negative"] & self.acceleration["Stable"]),
            consequent=(self.control_intent["Light_Brake"] % 1.0),
            label="rule_8",
            and_func=np.fmin,
            or_func=np.fmax,
        )
        self._rule9 = ctrl.Rule(
            antecedent=(self.speed_error["Negative"] & self.acceleration["Accelerate"]),
            consequent=(self.control_intent["Strong_Brake"] % 1.0),
            label="rule_9",
            and_func=np.fmin,
            or_func=np.fmax,
        )

        # --------------- Controller Assembly ---------------
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
        # Enable caching and bounds clipping for runtime stability
        self.control_simulation = ctrl.ControlSystemSimulation(
            control_system=self.control_system,
            clip_to_bounds=True,
            cache=True,
            flush_after_run=10,
            lenient=True,
        )

    def compute_control_intent(self, speed_error: float, acceleration: float) -> float:
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
        command_contribution = self._inspect_intent_contribution(command_value)

        # Map command value to CARLA throttle and brake controls
        if command_value >= 0:
            throttle = command_value
            brake = 0.0
        else:
            throttle = 0.0
            # Cap brake for comfort
            brake = min(0.3, abs(command_value))
            # brake = abs(command_value)

        throttle_brake_command = {"throttle": throttle, "brake": brake}

        return throttle_brake_command, command_contribution

    def _inspect_intent_contribution(self, intent_value: float) -> Dict[str, float]:
        """
        Return membership strengths of an output value across all consequent terms.

        Args:
            intent_value (float): Crisp output value produced by `compute_control_intent`.

        Returns:
            active_membership_terms (Dict[str, float]): Mapping of term labels to their non-zero membership strengths.
        """
        active_membership_terms = {}

        # Iterate over all consequent terms defined for the output variable
        for label in self.control_intent.terms:
            term = self.control_intent[label]

            # Compute degree of membership of intent_value in the current term
            degree = fuzz.interp_membership(
                x=self.control_intent.universe,
                xmf=term.mf,
                xx=intent_value,
                zero_outside_x=False,
            )

            clean_degree = round(float(degree), 2)

            # Record only terms that contribute meaningfully
            if clean_degree > 0.0:
                clean_label = label.replace("_", " ")
                active_membership_terms[clean_label] = clean_degree

        return active_membership_terms

    def visualize_system(self) -> None:
        """
        Render all visual components of the fuzzy control system.

        This method displays:
        - Antecedent and consequent membership functions.
        - The overall fuzzy control system structure.
        - (Optional) individual rule visualizations, which can be enabled as needed.
        """
        # Plot antecedent and consequent membership functions with simulation context
        self.speed_error.view(sim=self.control_simulation)
        plt.show()
        self.acceleration.view(sim=self.control_simulation)
        plt.show()
        self.control_intent.view(sim=self.control_simulation)
        plt.show()

        # Optional: enable these to inspect individual fuzzy rules
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

        # Plot aggregated fuzzy control system views
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
    flc.visualize_system()
    flc.control_simulation.inputs({"speed_error": 0.0, "acceleration": 0.0})
    flc.control_simulation.compute()
    print(flc.control_simulation.output["control_intent"])
    flc.speed_error.view(sim=flc.control_simulation)
    plt.show()
    flc.acceleration.view(sim=flc.control_simulation)
    plt.show()
    flc.control_intent.view(sim=flc.control_simulation)
    plt.show()
    flc.compute_control_intent(0.0, 0.0)
