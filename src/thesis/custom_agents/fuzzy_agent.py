import logging

import carla
from agents.navigation.basic_agent import BasicAgent

# Retrieve the module-level logger
LOGGER = logging.getLogger(__name__)


class FuzzyAgent(BasicAgent):
    def __init__(self, vehicle, target_speed=20, opt_dict={}, debug=False):
        super().__init__(vehicle, target_speed, opt_dict, debug)

    def run_step(self):
        control = carla.VehicleControl()
        return control
