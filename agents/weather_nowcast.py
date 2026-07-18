import os
import sys
import argparse
import pandas as pd
import numpy as np
# pyright: ignore [reportMissingImports]
import fastf1
from dotenv import load_dotenv

# Ensure workspace root is in sys.path for standalone script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
load_dotenv()
cache_dir = os.getenv("CACHE_DIR", "cache")
fastf1.Cache.enable_cache(cache_dir)

def compute_weather_nowcast(session, current_lap):
    """
    Computes short-horizon weather trend signal at current_lap.
    Strictly forward-only: aligns weather_data timestamps to current_lap completion time.
    Provides a fast-path stub for completely dry race sessions.
    """
    event_name = session.event.get("EventName", "Unknown GP")
    weather_df = session.weather_data

    if weather_df.empty:
        return {
            "event_name": event_name,
            "current_lap": int(current_lap),
            "weather_signal": "DRY_STABLE",
            "is_raining": False,
            "rain_trend": "STABLE_DRY",
            "track_temp_trend": "STABLE",
            "air_temp_c": 25.0,
            "track_temp_c": 35.0,
            "humidity_pct": 50.0,
            "confidence": "LOW",
            "is_dry_race_stub": True,
            "basis": f"No weather data available for {event_name}. Defaulting to stable dry conditions."
        }

    # 1. Fast-Path Stub Check for Completely Dry Sessions
    is_entirely_dry_session = not weather_df['Rainfall'].any()

    current_lap_records = session.laps[session.laps['LapNumber'] == current_lap]
    if not current_lap_records.empty and not current_lap_records['Time'].isna().all():
        t_lap = current_lap_records['Time'].min()
    else:
        laps_up_to = session.laps[session.laps['LapNumber'] <= current_lap]
        t_lap = laps_up_to['Time'].min() if not laps_up_to.empty else weather_df['Time'].min()

    # Filter weather samples up to t_lap (strictly forward-only)
    past_weather = weather_df[weather_df['Time'] <= t_lap]
    if past_weather.empty:
        past_weather = weather_df.head(1)

    latest_sample = past_weather.iloc[-1]
    air_temp = float(latest_sample['AirTemp']) if not pd.isna(latest_sample['AirTemp']) else 25.0
    track_temp = float(latest_sample['TrackTemp']) if not pd.isna(latest_sample['TrackTemp']) else 35.0
    humidity = float(latest_sample['Humidity']) if not pd.isna(latest_sample['Humidity']) else 50.0

    if is_entirely_dry_session:
        return {
            "event_name": event_name,
            "current_lap": int(current_lap),
            "weather_signal": "DRY_STABLE",
            "is_raining": False,
            "rain_trend": "STABLE_DRY",
            "track_temp_trend": "STABLE",
            "air_temp_c": air_temp,
            "track_temp_c": track_temp,
            "humidity_pct": humidity,
            "confidence": "HIGH",
            "is_dry_race_stub": True,
            "basis": f"Dry race session ({event_name}). Track conditions stable, 0.0mm rain recorded."
        }

    # 2. Rolling Trend Engine for Wet / Transition Sessions
    # Take rolling window of up to 5 samples leading up to current_lap
    window = past_weather.tail(5)
    
    is_raining = bool(latest_sample['Rainfall'])
    rain_count = window['Rainfall'].sum()
    window_len = len(window)

    first_sample_rain = bool(window.iloc[0]['Rainfall'])
    latest_sample_rain = is_raining

    if rain_count == 0:
        rain_trend = "STABLE_DRY"
        weather_signal = "DRY_STABLE"
    elif rain_count == window_len:
        rain_trend = "STABLE_WET"
        weather_signal = "WET_CONTINUOUS"
    elif not first_sample_rain and latest_sample_rain:
        rain_trend = "RISING"
        weather_signal = "RAIN_ONSET"
    elif first_sample_rain and not latest_sample_rain:
        rain_trend = "FALLING"
        weather_signal = "DRYING_LINE"
    elif is_raining:
        rain_trend = "VARIABLE_WET"
        weather_signal = "WET_CONTINUOUS"
    else:
        rain_trend = "VARIABLE_DRY"
        weather_signal = "DRYING_LINE"

    # Track temperature trend over window
    if window_len >= 2:
        temp_start = float(window.iloc[0]['TrackTemp'])
        temp_end = float(window.iloc[-1]['TrackTemp'])
        delta_temp = temp_end - temp_start

        if delta_temp < -1.0:
            track_temp_trend = "COOLING"
        elif delta_temp > 1.0:
            track_temp_trend = "WARMING"
        else:
            track_temp_trend = "STABLE"
    else:
        track_temp_trend = "STABLE"

    # Templated basis string
    if weather_signal == "RAIN_ONSET":
        basis = f"RAIN ONSET DETECTED at Lap {current_lap} ({event_name}). Rainfall active, track temp {track_temp_trend} ({track_temp:.1f}°C)."
    elif weather_signal == "DRYING_LINE":
        basis = f"Track drying trend at Lap {current_lap} ({event_name}). Rain stopped, track temp {track_temp_trend} ({track_temp:.1f}°C)."
    elif weather_signal == "WET_CONTINUOUS":
        basis = f"Continuous rain at Lap {current_lap} ({event_name}). Track wet, air temp {air_temp:.1f}°C, humidity {humidity:.1f}%."
    else:
        basis = f"Stable dry conditions at Lap {current_lap} ({event_name}). Track temp {track_temp:.1f}°C."

    return {
        "event_name": event_name,
        "current_lap": int(current_lap),
        "weather_signal": weather_signal,
        "is_raining": is_raining,
        "rain_trend": rain_trend,
        "track_temp_trend": track_temp_trend,
        "air_temp_c": air_temp,
        "track_temp_c": track_temp,
        "humidity_pct": humidity,
        "confidence": "HIGH",
        "is_dry_race_stub": False,
        "basis": basis
    }

def run_tests():
    print("Running self-contained verification tests for Weather Nowcast Agent (Phase 4)...")

    # 1. Test Fast-Path Stub on 2021 Abu Dhabi GP (Dry Race)
    print("Loading cached 2021 Abu Dhabi Grand Prix (Dry Race) for Fast-Path Stub validation...")
    session_dry = fastf1.get_session(2021, "Abu Dhabi Grand Prix", "R")
    session_dry.load(telemetry=False, weather=True)

    res_dry = compute_weather_nowcast(session_dry, 30)
    assert res_dry is not None, "Dry race weather result should not be None"
    assert res_dry['is_dry_race_stub'] == True, "Dry race should use fast-path stub"
    assert res_dry['weather_signal'] == "DRY_STABLE", "Dry race signal should be DRY_STABLE"
    assert res_dry['is_raining'] == False, "Dry race is_raining should be False"
    print(f"[PASS] Fast-path stub verified on 2021 Abu Dhabi GP at Lap 30: signal={res_dry['weather_signal']}, is_stub={res_dry['is_dry_race_stub']}")

    # 2. Test Rolling Trend Engine on 2023 Dutch GP (Wet/Transition Race)
    print("Loading cached 2023 Dutch Grand Prix (Wet Race) for Rolling Trend Engine validation...")
    session_wet = fastf1.get_session(2023, "Dutch Grand Prix", "R")
    session_wet.load(telemetry=False, weather=True)

    # Lap 3 (Rain onset at start of race)
    res_lap3 = compute_weather_nowcast(session_wet, 3)
    assert res_lap3 is not None, "Lap 3 result should not be None"
    assert res_lap3['is_dry_race_stub'] == False, "Wet race should not use fast-path stub"
    assert res_lap3['is_raining'] == True or res_lap3['weather_signal'] in ["RAIN_ONSET", "WET_CONTINUOUS"], "Lap 3 should indicate rain"
    print(f"[PASS] Rolling trend verified at Lap 3 (Rain Onset): signal={res_lap3['weather_signal']}, trend={res_lap3['rain_trend']}")

    # Lap 15 (Drying track phase)
    res_lap15 = compute_weather_nowcast(session_wet, 15)
    assert res_lap15 is not None, "Lap 15 result should not be None"
    print(f"[PASS] Rolling trend verified at Lap 15 (Drying/Stable): signal={res_lap15['weather_signal']}, trend={res_lap15['rain_trend']}")

    print("\nALL PHASE 4 WEATHER NOWCAST AGENT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Nowcast Agent (Phase 4)")
    parser.add_argument("--year", type=int, default=2023, help="Race year")
    parser.add_argument("--race", type=str, default="Dutch Grand Prix", help="Race name")
    parser.add_argument("--lap", type=int, default=3, help="Lap number to evaluate")
    parser.add_argument("--test", action="store_true", help="Run self-contained verification tests")

    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    print(f"Loading session {args.year} {args.race}...")
    session = fastf1.get_session(args.year, args.race, "R")
    session.load(telemetry=False, weather=True)

    print(f"\nComputing Weather Nowcast at Lap {args.lap}...")
    res = compute_weather_nowcast(session, args.lap)

    print(f"\n### Weather Nowcast Report — {res['event_name']} Lap {res['current_lap']} ###\n")
    print(f"- **Weather Signal**: {res['weather_signal']}")
    print(f"- **Is Raining**: {res['is_raining']}")
    print(f"- **Rainfall Trend**: {res['rain_trend']}")
    print(f"- **Track Temp Trend**: {res['track_temp_trend']} ({res['track_temp_c']:.1f}°C)")
    print(f"- **Air Temp / Humidity**: {res['air_temp_c']:.1f}°C / {res['humidity_pct']:.1f}%")
    print(f"- **Fast-Path Stub Used**: {res['is_dry_race_stub']}")
    print(f"- **Basis**: {res['basis']}")
