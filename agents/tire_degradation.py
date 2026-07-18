import os
import sys
import argparse
import pandas as pd
import numpy as np
# pyright: ignore [reportMissingImports]
import fastf1
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
cache_dir = os.getenv("CACHE_DIR", "cache")
fastf1.Cache.enable_cache(cache_dir)

def fuel_corrected_time(raw_time_seconds, lap_number, total_laps, fuel_effect_per_lap=0.075):
    """
    Corrects a raw lap time in seconds to remove the effect of fuel burn-off.
    We correct to a 'zero-fuel' (end of race) baseline.
    As the race progresses, the car gets lighter and faster.
    To compare a lap time at lap L to the end of the race, we subtract the fuel weight penalty.
    """
    if pd.isna(raw_time_seconds):
        return np.nan
    fuel_penalty = (total_laps - lap_number) * fuel_effect_per_lap
    return raw_time_seconds - fuel_penalty

def fit_degradation_curve(clean_laps, compound):
    """
    Fits a linear regression model to fuel-corrected lap times against tire life.
    y = beta_0 + beta_1 * x
    where x = TyreLife, y = FuelCorrectedLapTime
    Returns: (slope, intercept)
    """
    default_priors = {
        'SOFT': 0.06,
        'MEDIUM': 0.04,
        'HARD': 0.025,
        'INTERMEDIATE': 0.05,
        'WET': 0.07
    }
    compound_key = str(compound).upper()
    default_slope = default_priors.get(compound_key, 0.04)

    if len(clean_laps) < 3:
        # Fallback to priors if there are too few clean laps
        if not clean_laps.empty:
            avg_x = clean_laps['TyreLife'].mean()
            avg_y = clean_laps['FuelCorrectedLapTime'].mean()
            intercept = avg_y - default_slope * avg_x
        else:
            intercept = 90.0  # Arbitrary baseline
        return default_slope, intercept
    else:
        try:
            # Fit a line
            slope, intercept = np.polyfit(clean_laps['TyreLife'], clean_laps['FuelCorrectedLapTime'], 1)
            # Clip slope to a small positive floor to represent minimal wear if it fits negative
            # (which can happen due to noise or rapid track evolution)
            if slope < 0.005:
                slope = 0.005
            return slope, intercept
        except Exception:
            return default_slope, 90.0

def compute_driver_degradation(session, driver, current_lap, fuel_effect_per_lap=0.075):
    """
    Computes tire degradation metrics for a single driver up to current_lap.
    Strictly forward-only: does not look at laps > current_lap.
    """
    try:
        # Use pick_drivers (non-deprecated replacement for pick_driver)
        driver_laps = session.laps.pick_drivers(driver)
    except Exception:
        return None

    # Filter for completed laps up to the current lap
    laps_up_to_current = driver_laps[driver_laps['LapNumber'] <= current_lap]
    if laps_up_to_current.empty:
        return None

    # Get the driver's current lap record
    current_lap_record = laps_up_to_current[laps_up_to_current['LapNumber'] == current_lap]
    if current_lap_record.empty:
        return None
    current_lap_record = current_lap_record.iloc[0]

    current_stint = current_lap_record['Stint']
    current_compound = current_lap_record['Compound']
    current_tyre_life = current_lap_record['TyreLife']

    # Select all laps in this stint up to the current lap
    stint_laps = laps_up_to_current[laps_up_to_current['Stint'] == current_stint]

    # Calculate fuel-corrected times for all laps in this stint
    total_laps = session.laps['LapNumber'].max()
    stint_laps_corrected = stint_laps.copy()
    
    # Calculate float seconds from timedelta64
    stint_laps_corrected['RawLapTimeSec'] = stint_laps_corrected['LapTime'].apply(
        lambda t: t.total_seconds() if not pd.isna(t) else np.nan
    )
    stint_laps_corrected['FuelCorrectedLapTime'] = stint_laps_corrected.apply(
        lambda r: fuel_corrected_time(r['RawLapTimeSec'], r['LapNumber'], total_laps, fuel_effect_per_lap),
        axis=1
    )

    # Filter clean laps (exclude SC/VSC, pit entries/exits, and NaT times)
    clean_laps = stint_laps_corrected[
        (stint_laps_corrected['FuelCorrectedLapTime'].notna()) &
        (stint_laps_corrected['TrackStatus'].astype(str) == '1') &
        (stint_laps_corrected['PitInTime'].isna()) &
        (stint_laps_corrected['PitOutTime'].isna())
    ]

    # Remove outlier laps to protect the fit (if we have enough laps)
    if len(clean_laps) >= 3:
        q1 = clean_laps['FuelCorrectedLapTime'].quantile(0.25)
        q3 = clean_laps['FuelCorrectedLapTime'].quantile(0.75)
        iqr = q3 - q1
        clean_laps = clean_laps[
            (clean_laps['FuelCorrectedLapTime'] >= q1 - 1.5 * iqr) &
            (clean_laps['FuelCorrectedLapTime'] <= q3 + 1.5 * iqr)
        ]

    # Fit degradation curve
    slope, intercept = fit_degradation_curve(clean_laps, current_compound)

    # Calculate pace delta vs fresh (TyreLife = 1)
    # pace_delta = fitted_pace_at_current_life - fitted_pace_at_life_1
    # since y = slope * x + intercept, pace_delta = slope * (current_tyre_life - 1)
    pace_delta = max(0.0, slope * (current_tyre_life - 1))

    # Normalize into a degradation score (0.0 to 1.0)
    # Define expected max pace loss thresholds per compound before a pit is highly urgent
    thresholds = {
        'SOFT': 2.0,
        'MEDIUM': 1.8,
        'HARD': 1.5,
        'INTERMEDIATE': 2.0,
        'WET': 2.5
    }
    max_loss = thresholds.get(str(current_compound).upper(), 1.8)
    degradation_score = min(1.0, pace_delta / max_loss)

    # Derive pit urgency
    if degradation_score < 0.4:
        pit_urgency = "LOW"
    elif degradation_score < 0.7:
        pit_urgency = "MEDIUM"
    else:
        pit_urgency = "HIGH"

    # Templated basis string (no LLM reasoning)
    basis = (
        f"Tire age is {int(current_tyre_life)} laps on {current_compound}. "
        f"Pace loss is +{pace_delta:.2f}s vs fresh (degradation rate is {slope*1000:.1f}ms/lap)."
    )

    # Deriving trend for next 3 laps
    pace_delta_3_laps = max(0.0, slope * (current_tyre_life + 3 - 1))
    deg_score_3_laps = min(1.0, pace_delta_3_laps / max_loss)
    
    pace_delta_5_laps = max(0.0, slope * (current_tyre_life + 5 - 1))
    deg_score_5_laps = min(1.0, pace_delta_5_laps / max_loss)

    if deg_score_3_laps - degradation_score < 0.05:
        trend_3_laps = "flat/stable"
    elif deg_score_3_laps >= 0.7 and degradation_score < 0.7:
        trend_3_laps = "crossing_into_high_degradation"
    else:
        trend_3_laps = "accelerating_degradation"

    raw_lap_sec = stint_laps_corrected.loc[stint_laps_corrected['LapNumber'] == current_lap, 'RawLapTimeSec'].values[0]
    corr_lap_sec = stint_laps_corrected.loc[stint_laps_corrected['LapNumber'] == current_lap, 'FuelCorrectedLapTime'].values[0]

    return {
        'driver': driver,
        'stint': int(current_stint),
        'compound': current_compound,
        'tyre_life': int(current_tyre_life),
        'raw_lap_time': raw_lap_sec if not pd.isna(raw_lap_sec) else None,
        'corrected_lap_time': corr_lap_sec if not pd.isna(corr_lap_sec) else None,
        'deg_rate_ms_per_lap': slope * 1000.0,
        'pace_delta_vs_fresh': pace_delta,
        'degradation_score': degradation_score,
        'pit_urgency': pit_urgency,
        'lookahead_degradation_score_next_3_laps': deg_score_3_laps,
        'lookahead_degradation_score_next_5_laps': deg_score_5_laps,
        'lookahead_trend_next_3_laps': trend_3_laps,
        'basis': basis,
        # Keep clean laps for plotting access if needed
        '_clean_laps_df': clean_laps,
        '_stint_laps_df': stint_laps_corrected,
        '_slope': slope,
        '_intercept': intercept
    }

def get_field_degradation(session, current_lap, fuel_effect_per_lap=0.075):
    """
    Computes tire degradation metrics for all active drivers at current_lap.
    """
    results = []
    # Find all unique drivers in the race
    drivers = session.laps['Driver'].unique()
    for driver in drivers:
        deg = compute_driver_degradation(session, driver, current_lap, fuel_effect_per_lap)
        if deg:
            results.append(deg)
    return results

def plot_driver_stint(deg_data, output_dir="temp"):
    """
    Generates and saves a visualization plot of the driver's tire stint.
    """
    # pyright: ignore [reportMissingImports]
    import matplotlib.pyplot as plt
    
    driver = deg_data['driver']
    stint = deg_data['stint']
    compound = deg_data['compound']
    clean_laps = deg_data['_clean_laps_df']
    stint_laps = deg_data['_stint_laps_df']
    slope = deg_data['_slope']
    intercept = deg_data['_intercept']

    if stint_laps.empty:
        print(f"No laps to plot for driver {driver}.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw vs fuel-corrected times
    plt.scatter(stint_laps['TyreLife'], stint_laps['RawLapTimeSec'], color='gray', alpha=0.5, label='Raw Lap Time (seconds)')
    plt.scatter(stint_laps['TyreLife'], stint_laps['FuelCorrectedLapTime'], color='blue', marker='o', label='Fuel-Corrected Lap Time')
    
    # Highlight clean laps used for fitting
    if not clean_laps.empty:
        plt.scatter(clean_laps['TyreLife'], clean_laps['FuelCorrectedLapTime'], color='green', marker='x', s=100, label='Clean Laps (used for fit)')

    # Plot fitted curve
    x_fit = np.linspace(1, stint_laps['TyreLife'].max() + 1, 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Fitted Curve (slope={slope*1000:.1f}ms/lap)')
    
    plt.title(f"Tire Degradation Fit: {driver} - Stint {stint} ({compound})")
    plt.xlabel("Tire Life (Laps)")
    plt.ylabel("Lap Time (Seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filename = os.path.join(output_dir, f"degradation_{driver}_stint_{stint}.png")
    plt.savefig(filename)
    plt.close()
    print(f"\n[PLOT SUCCESS] Saved degradation plot to: {filename}")

def run_tests():
    print("Running self-contained verification tests...")
    
    # 1. Fuel correction test
    t_corr = fuel_corrected_time(100.0, 10, 50, 0.1)
    assert abs(t_corr - 96.0) < 1e-6, f"Fuel correction failed: expected 96.0, got {t_corr}"
    print("[PASS] Fuel correction math is correct.")

    # 2. Curve fitting prior test (short stint)
    df_short = pd.DataFrame({'TyreLife': [1.0, 2.0], 'FuelCorrectedLapTime': [80.0, 80.05]})
    slope, intercept = fit_degradation_curve(df_short, 'SOFT')
    assert abs(slope - 0.06) < 1e-6, f"Fallback prior for SOFT failed: expected 0.06, got {slope}"
    print("[PASS] Fallback prior for short stints behaves correctly.")

    # 3. Curve fitting linear fit test
    df_fit = pd.DataFrame({'TyreLife': [1.0, 2.0, 3.0, 4.0], 'FuelCorrectedLapTime': [80.0, 80.05, 80.10, 80.15]})
    slope, intercept = fit_degradation_curve(df_fit, 'MEDIUM')
    assert abs(slope - 0.05) < 1e-6, f"Regression slope failed: expected 0.05, got {slope}"
    print("[PASS] Regression curve fitting fits exact linear slopes correctly.")
    
    # 4. End-to-end load test on cached 2023 Dutch GP
    print("Loading cached 2023 Dutch Grand Prix for E2E validation...")
    session = fastf1.get_session(2023, "Dutch Grand Prix", "R")
    session.load(telemetry=False, weather=False)
    
    # Check Verstappen's degradation at lap 40
    ver_deg = compute_driver_degradation(session, "VER", 40)
    assert ver_deg is not None, "Verstappen degradation data should not be None at Lap 40."
    assert ver_deg['compound'] == 'SOFT', f"Expected compound SOFT, got {ver_deg['compound']}"
    assert ver_deg['tyre_life'] == 29, f"Expected tyre life 29, got {ver_deg['tyre_life']}"
    assert ver_deg['degradation_score'] >= 0.0 and ver_deg['degradation_score'] <= 1.0, f"Expected score in [0, 1], got {ver_deg['degradation_score']}"
    assert ver_deg['pit_urgency'] in ["LOW", "MEDIUM", "HIGH"], f"Invalid pit urgency: {ver_deg['pit_urgency']}"
    print(f"[PASS] E2E validation passed. VER Lap 40: score={ver_deg['degradation_score']:.3f}, urgency={ver_deg['pit_urgency']}")
    
    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tire Degradation Agent (Phase 1)")
    parser.add_argument("--year", type=int, default=2023, help="Race year")
    parser.add_argument("--race", type=str, default="Dutch Grand Prix", help="Race name")
    parser.add_argument("--lap", type=int, default=40, help="Lap number to evaluate")
    parser.add_argument("--plot", type=str, default=None, help="Driver code to plot current stint for (e.g. VER)")
    parser.add_argument("--test", action="store_true", help="Run self-contained verification tests")
    
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    print(f"Loading session {args.year} {args.race}...")
    session = fastf1.get_session(args.year, args.race, "R")
    session.load(telemetry=False, weather=False)

    print(f"\nComputing degradation field-wide at Lap {args.lap}...")
    field_deg = get_field_degradation(session, args.lap)

    # Convert to DataFrame for pretty printing
    df_show = pd.DataFrame(field_deg)
    if not df_show.empty:
        # Drop internal fields for cleaner printing
        cols_to_print = [
            'driver', 'stint', 'compound', 'tyre_life', 
            'raw_lap_time', 'corrected_lap_time', 'deg_rate_ms_per_lap', 
            'pace_delta_vs_fresh', 'degradation_score', 'pit_urgency'
        ]
        
        # Sort by degradation score descending
        df_show = df_show[cols_to_print].sort_values(by='degradation_score', ascending=False)
        
        print(f"\n### Tire Degradation Report — Lap {args.lap} ###\n")
        print(df_show.to_markdown(index=False))
    else:
        print("No active drivers found or data not available at this lap.")

    # Handle plotting request
    if args.plot:
        driver_code = args.plot.upper()
        driver_deg = next((d for d in field_deg if d['driver'] == driver_code), None)
        if driver_deg:
            plot_driver_stint(driver_deg)
        else:
            print(f"Could not find degradation data for driver {driver_code} to plot.")
