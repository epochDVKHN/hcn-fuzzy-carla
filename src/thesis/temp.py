"""
Module: HumanCenteredCruise
Thesis Context: Tier 3 (Context Evaluator) - Atomic Behavior
Architecture: Embedded Physics (Direct Mapping).
    - Logic: Human-Centered (Lazy Braking / Smoothness).
    - Physics: Non-Linear Plant dynamics encoded directly in Consequent MFs.
"""

import carla
import numpy as np
import skfuzzy as fuzz
from agents.navigation.basic_agent import BasicAgent
from skfuzzy import control as ctrl

# ==============================================================================
# 1. THE FUZZY LOGIC CONTROLLER
# ==============================================================================

class FLC_FreeFlow:
    def __init__(self, target_speed_kmh=30.0):
        self.target_speed = target_speed_kmh / 3.6
        self.simulation = None
        self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        # --- INPUTS (LAZY / HUMAN-CENTERED) ---

        # Speed Error [km/h]
        self.e_speed = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'error_speed')

        # Negative: Lazy braking (-3.0 start).
        # "I'm 2 km/h over? I'll just coast."
        self.e_speed['neg']  = fuzz.trapmf(self.e_speed.universe, [-10, -10, -3.0, 0.0])

        # Zero: Wide stability.
        self.e_speed['zero'] = fuzz.trimf(self.e_speed.universe,  [-3.0, 0.0, 2.0])

        # Positive: Standard response.
        self.e_speed['pos']  = fuzz.trapmf(self.e_speed.universe, [0.0, 2.0, 10.0, 10.0])

        # Acceleration [m/s^2]
        self.acceleration = ctrl.Antecedent(np.arange(-3, 3.1, 0.1), 'acceleration')
        self.acceleration['decel']  = fuzz.trapmf(self.acceleration.universe, [-3, -3, -1.5, 0])
        self.acceleration['stable'] = fuzz.trimf(self.acceleration.universe,  [-1.5, 0, 1.0])
        self.acceleration['accel']  = fuzz.trapmf(self.acceleration.universe, [0.0, 1.0, 3, 3])

        # --- OUTPUT (EMBEDDED PHYSICS) ---
        # Universe: [-1, 1] (Negative=Brake, Positive=Throttle)
        self.control = ctrl.Consequent(np.arange(-1, 1.01, 0.01), 'control')
        self.control.defuzzify_method = 'centroid'

        # 1. Strong Brake
        self.control['brake_strong'] = fuzz.trapmf(self.control.universe, [-1, -1, -0.5, -0.05])

        # 2. Light Brake (Engine Drag)
        # Peak at +0.15 Throttle. Uses drag to slow down.
        self.control['brake_light']  = fuzz.trimf(self.control.universe,  [-0.05, 0.15, 0.45])

        # 3. Maintain (City Trim)
        # Peak at +0.45 Throttle. Holds ~20-25 km/h.
        self.control['maintain']     = fuzz.trimf(self.control.universe,  [0.15, 0.45, 0.60])

        # 4. Light Accel (Highway Trim)
        # Peak at +0.60 Throttle. Holds ~90 km/h.
        self.control['accel_light']  = fuzz.trimf(self.control.universe,  [0.45, 0.60, 0.80])

        # 5. Strong Accel (Max Power)
        self.control['accel_strong'] = fuzz.trapmf(self.control.universe, [0.60, 0.80, 1, 1])

        # --- RULES ---
        rules = []

        # Too Slow
        rules.append(ctrl.Rule(self.e_speed['pos'] & self.acceleration['decel'],  self.control['accel_strong']))
        rules.append(ctrl.Rule(self.e_speed['pos'] & self.acceleration['stable'], self.control['accel_strong']))
        rules.append(ctrl.Rule(self.e_speed['pos'] & self.acceleration['accel'],  self.control['maintain'])) # Damping

        # On Target
        rules.append(ctrl.Rule(self.e_speed['zero'] & self.acceleration['decel'],  self.control['accel_light'])) # Fight Drag
        rules.append(ctrl.Rule(self.e_speed['zero'] & self.acceleration['stable'], self.control['maintain']))
        rules.append(ctrl.Rule(self.e_speed['zero'] & self.acceleration['accel'],  self.control['brake_light'])) # Gravity Check

        # Too Fast
        rules.append(ctrl.Rule(self.e_speed['neg'] & self.acceleration['decel'],  self.control['brake_light'])) # Coast (Throttle 0.15)
        rules.append(ctrl.Rule(self.e_speed['neg'] & self.acceleration['stable'], self.control['brake_light']))
        rules.append(ctrl.Rule(self.e_speed['neg'] & self.acceleration['accel'],  self.control['brake_strong'])) # Panic

        self.ctrl_system = ctrl.ControlSystem(rules)
        # Cache=False is safer for continuous inputs
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system, clip_to_bounds=True, cache=False)

    def compute(self, current_speed_ms, current_accel_ms2):
        error_ms = self.target_speed - current_speed_ms
        error_kmh = error_ms * 3.6

        # Clamping protects against outliers
        e_in = max(-10, min(10, error_kmh))
        a_in = max(-3, min(3, current_accel_ms2))

        self.simulation.input['error_speed'] = e_in
        self.simulation.input['acceleration'] = a_in
        try:
            self.simulation.compute()
            return self.simulation.output['control']
        except:
            return 0.0

# ==============================================================================
# 2. AGENT WRAPPER
# ==============================================================================

class HumanAgent(BasicAgent):
    def __init__(self, vehicle, target_speed=20):
        super().__init__(vehicle, target_speed)
        self.flc = FLC_FreeFlow(target_speed_kmh=target_speed)

    def run_step(self):
        # 1. Perception (Ground Truth via API)
        velocity_vec = self._vehicle.get_velocity()
        accel_vec = self._vehicle.get_acceleration()
        fwd_vec = self._vehicle.get_transform().get_forward_vector()

        # Calculate Scalars using built-in vector math
        current_speed = 3.6 * velocity_vec.length()
        accel = accel_vec.dot(fwd_vec) # Project world accel to longitudinal

        # 2. Compute FLC
        control_val = self.flc.compute(current_speed, accel)

        # 3. Actuation (Direct Physics Pass-Through)
        control = carla.VehicleControl()

        if control_val >= 0:
            control.throttle = float(control_val)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            # Cap brake at 0.3 for comfort (verified by logs)
            control.brake = min(0.3, abs(float(control_val)))

        return control

    self.control['brake_strong'] = fuzz.trapmf(self.control.universe, [-1, -1, -0.6, -0.3])

        # 2. Light Brake (Light Friction / Drag Assist)
        # Centered at -0.3. Maps to Brake 0.3 directly.
        self.control['brake_light']  = fuzz.trimf(self.control.universe,  [-0.6, -0.3, 0.0])

        # 3. Maintain (City Trim)
        # Centered at +0.45 Throttle (Matches log data for ~20-30 km/h)
        self.control['maintain']     = fuzz.trimf(self.control.universe,  [-0.3, 0.45, 0.60])

        # 4. Light Accel (Highway Trim)
        # Centered at +0.60 Throttle (Matches log data for ~90 km/h)
        self.control['accel_light']  = fuzz.trimf(self.control.universe,  [0.45, 0.60, 0.80])

        # 5. Strong Accel (Max Power)
        self.control['accel_strong'] = fuzz.trapmf(self.control.universe, [0.60, 0.80, 1, 1])

def _update_target_speed(self) -> None:
        CURVE_CHECK = self._check_road_curvature()
        current_speed = get_speed(self._vehicle)

        if CURVE_CHECK:
            # We are in a curve.

            # 1. Have we latched a speed yet? If not, grab it NOW.
            if self._curve_entry_speed is None:
                self._curve_entry_speed = current_speed

            # 2. Determine the limit
            # We take the SMALLER of:
            #  a) The hard 30km/h limit
            #  b) The speed we had when we entered the turn
            target_limit = min(30.0, self._curve_entry_speed)

            # Set the target. This value is now FIXED for the duration of the turn.
            # This prevents the "deceleration spiral."
            self.set_target_speed(target_limit)

        else:
            # We are on a straight.

            # 1. Reset the latch so it's ready for the next turn
            self._curve_entry_speed = None

            # 2. Resume normal speed limit
            self.set_target_speed(self._vehicle.get_speed_limit())

def inspect_output_contribution(self, final_output):
        """
        Visualizes which Consequent Membership Functions describe the final output.
        Does NOT re-compute the fuzzy system (Efficiency Optimization).

        Args:
            final_output (float): The crisp output value from compute_control_command.

        Returns:
            dict: Active terms and their membership strength (0.0 to 1.0).
        """
        print(f"\n--- Output Inspector (Value: {final_output:.3f}) ---")

        active_terms = {}
        total_membership = 0.0

        # Iterate through all defined output terms (Strong Brake, Maintain, etc.)
        # We access 'self.control' which is already stored in the class instance.
        for label in self.control.terms:
            term = self.control[label]

            # Check: How much does this specific output value belong to this Triangle/Trapezoid?
            # We use the Universe of Discourse and the MF shape (mf) stored in the term.
            degree = fuzz.interp_membership(self.control.universe, term.mf, final_output)

            # Filter out zero/negligible contributions
            if degree > 0.001:
                active_terms[label] = degree
                total_membership += degree

        # Print/Return Results
        # If total > 0, we normalize to percentages for readability
        if total_membership > 0:
            for label, degree in active_terms.items():
                percent = (degree / total_membership) * 100
                print(f"  {label}: {percent:.1f}% (Raw: {degree:.2f})")
        else:
             print("  (Output is in a dead zone or undefined)")

        print("------------------------------------------------")
        return

def _inspect_command_contribution(self, command_value: float):
        active_terms = {}
        total_membership = 0.0

        for label in self.control_intent.terms:
            term = self.control_intent[label]
            # zero_outside_x=False prevents floating point clipping errors at edges
            degree = fuzz.interp_membership(
                x=self.control_intent.universe,
                xmf=term.mf,
                xx=command_value,
                zero_outside_x=False,
            )

            if degree > 0.001: # Filter out 0.00000001 type noise
                # Convert np.float64 -> float and Round to 4 decimals
                clean_degree = round(float(degree), 4)

                # Optional: Clean up the label string (e.g. "Light_Brake" -> "Light Brake")
                clean_label = label.replace("_", " ")

                active_terms[clean_label] = clean_degree
                total_membership += degree

        # Optional: Print warning if math is broken (Validation)
        if not (0.99 < total_membership < 1.01):
            print(f"WARNING: Partition of Unity failed! Sum: {total_membership}")

        print(active_terms)
        return active_terms