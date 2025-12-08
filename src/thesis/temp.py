def _update_target_speed(self, CURVE_SPEED_LIMIT: float = 30.0) -> None:
        CURVE_CHECK = self._check_road_curvature()
        current_speed = get_speed(self._vehicle)

        if CURVE_CHECK:
            if self._curve_entry_speed is None:
                # --- THE FIX ---
                # Case A: We are stopped/spawning (approx 0 km/h).
                # We don't want to latch 0! We want to accelerate up to the limit.
                if current_speed < 2.0:
                    self._curve_entry_speed = CURVE_SPEED_LIMIT

                # Case B: We are already moving.
                # Latch the current speed to prevent the "deceleration spiral."
                else:
                    self._curve_entry_speed = current_speed

            # Set target to the minimum of the Hard Limit or our Latched Speed
            curve_target_speed_limit = round(
                min(CURVE_SPEED_LIMIT, self._curve_entry_speed), 0
            )
            self.set_target_speed(curve_target_speed_limit)

        else:
            self._curve_entry_speed = None
            # Resume normal speed limit on straights
            self.set_target_speed(self._vehicle.get_speed_limit())