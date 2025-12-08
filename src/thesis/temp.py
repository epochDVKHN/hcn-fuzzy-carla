# --- Table 5.1: Quantitative Performance Scorecard (Expanded) ---

# 1. Define Paths for Event Logs
baseline_col_path = SCENE_1_DIR / "baseline_collisions.csv"
baseline_inv_path = SCENE_1_DIR / "baseline_lane_invasions.csv"
fuzzy_col_path = SCENE_1_DIR / "fuzzy_collisions.csv"
fuzzy_inv_path = SCENE_1_DIR / "fuzzy_lane_invasions.csv"

# Helper for safety counts
def count_events(file_path):
    try:
        return len(pd.read_csv(file_path))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return 0

def calculate_scorecard(df_ts, col_path, inv_path, agent_name):
    """
    Computes all 11 metrics for a single agent.
    """
    # --- A. Mission Efficiency ---
    total_duration = df_ts['relative_time'].iloc[-1]
    avg_speed = df_ts['current_speed_kmh'].mean()

    # --- B. Tracking Accuracy ---
    velocity_error = df_ts['target_speed_kmh'] - df_ts['current_speed_kmh']
    rmse = np.sqrt((velocity_error ** 2).mean())

    # --- C. Ride Comfort ---
    avg_abs_jerk = df_ts['longitudinal_jerk'].abs().mean()
    peak_jerk = df_ts['longitudinal_jerk'].abs().max()

    # --- D. Vehicle Dynamics ---
    peak_speed = df_ts['current_speed_kmh'].max()
    peak_accel = df_ts['longitudinal_acceleration'].max()
    peak_decel = df_ts['longitudinal_acceleration'].min()

    # --- E. Control Stability ---
    control_std = df_ts['control_intent'].std()

    # --- F. Safety ---
    collision_count = count_events(col_path)
    invasion_count = count_events(inv_path)

    return {
        "Agent": agent_name,
        "Total Travel Time (s)": total_duration,
        "Average Velocity (km/h)": avg_speed,
        "Peak Velocity (km/h)": peak_speed,
        "Velocity RMSE (km/h)": rmse,
        "Avg Abs Jerk (m/s^3)": avg_abs_jerk,
        "Peak Jerk (m/s^3)": peak_jerk,
        "Peak Accel (m/s^2)": peak_accel,
        "Peak Decel (m/s^2)": peak_decel,
        "Control Std Dev": control_std,
        "Lane Invasions": invasion_count,
        "Collisions": collision_count
    }

# 2. Calculate Metrics
metrics_baseline = calculate_scorecard(df_baseline_ts, baseline_col_path, baseline_inv_path, "Baseline (PID)")
metrics_fuzzy = calculate_scorecard(df_fuzzy_ts, fuzzy_col_path, fuzzy_inv_path, "Fuzzy (HC-FLC)")

# 3. Format the Data Table
scorecard_data = [metrics_baseline, metrics_fuzzy]
df_scorecard = pd.DataFrame(scorecard_data)

# Transpose: Metrics as Rows, Agents as Columns
df_scorecard = df_scorecard.set_index("Agent").T

# 4. Rounding for Academic Presentation
# Mission Efficiency
df_scorecard.loc["Total Travel Time (s)"] = df_scorecard.loc["Total Travel Time (s)"].apply(lambda x: f"{x:.2f}")
df_scorecard.loc["Average Velocity (km/h)"] = df_scorecard.loc["Average Velocity (km/h)"].apply(lambda x: f"{x:.2f}")
df_scorecard.loc["Peak Velocity (km/h)"] = df_scorecard.loc["Peak Velocity (km/h)"].apply(lambda x: f"{x:.2f}")

# Tracking & Comfort
df_scorecard.loc["Velocity RMSE (km/h)"] = df_scorecard.loc["Velocity RMSE (km/h)"].apply(lambda x: f"{x:.2f}")
df_scorecard.loc["Avg Abs Jerk (m/s^3)"] = df_scorecard.loc["Avg Abs Jerk (m/s^3)"].apply(lambda x: f"{x:.2f}")
df_scorecard.loc["Peak Jerk (m/s^3)"] = df_scorecard.loc["Peak Jerk (m/s^3)"].apply(lambda x: f"{x:.2f}")

# Dynamics & Stability
df_scorecard.loc["Peak Accel (m/s^2)"] = df_scorecard.loc["Peak Accel (m/s^2)"].apply(lambda x: f"{x:.2f}")
df_scorecard.loc["Peak Decel (m/s^2)"] = df_scorecard.loc["Peak Decel (m/s^2)"].apply(lambda x: f"{x:.2f}")
df_scorecard.loc["Control Std Dev"] = df_scorecard.loc["Control Std Dev"].apply(lambda x: f"{x:.3f}")

# Safety
df_scorecard.loc["Lane Invasions"] = df_scorecard.loc["Lane Invasions"].astype(int)
df_scorecard.loc["Collisions"] = df_scorecard.loc["Collisions"].astype(int)

# 5. Display and Save
print("Table 5.1: Quantitative Performance Comparison")
display(df_scorecard)

# Export
save_path = PROJECT_ROOT / "results" / "tables" / "Table_5_1_Scorecard.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)
df_scorecard.to_csv(save_path)
print(f"\nScorecard saved to: {save_path}")

# LaTeX Export
# print("\nLaTeX Code:")
# print(df_scorecard.to_latex())