import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
# pyright: ignore [reportMissingImports]
import fastf1
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.tire_degradation import get_field_degradation
from agents.sc_probability import compute_sc_vsc_probability
from agents.rival_inference import infer_rival_strategy
from agents.weather_nowcast import compute_weather_nowcast

# Load environment variables
load_dotenv()
cache_dir = os.getenv("CACHE_DIR", "cache")
fastf1.Cache.enable_cache(cache_dir)

FIELD_STATES_CACHE_DIR = os.path.join(cache_dir, "field_states")

def sanitize_for_json(obj):
    """
    Recursively converts numpy numbers, pandas structures, and internal keys
    into standard JSON-serializable Python data structures.
    """
    if isinstance(obj, dict):
        clean_dict = {}
        for k, v in obj.items():
            # Exclude internal dataframe/fit references
            if str(k).startswith("_"):
                continue
            clean_dict[k] = sanitize_for_json(v)
        return clean_dict
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (pd.Timedelta, pd.Timestamp)):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def get_cache_filepath(year, race):
    """
    Returns path to precomputed field_state JSON file under cache/field_states/.
    """
    race_slug = str(race).lower().replace(" ", "_")
    os.makedirs(FIELD_STATES_CACHE_DIR, exist_ok=True)
    return os.path.join(FIELD_STATES_CACHE_DIR, f"field_state_{year}_{race_slug}.json")

def precompute_race_field_state(year, race, force_recompute=False):
    """
    Runs all rule-based agents (Phases 1-4) for every car and every lap of a race.
    Stores and serializes full field_state to disk.
    Returns: (field_state_dict, execution_time_sec)
    """
    filepath = get_cache_filepath(year, race)

    if not force_recompute and os.path.exists(filepath):
        print(f"[CACHE HIT] Loading precomputed field_state from: {filepath}")
        t0 = time.time()
        with open(filepath, "r") as f:
            data = json.load(f)
        elapsed = time.time() - t0
        print(f"[CACHE HIT] Loaded in {elapsed*1000:.2f}ms.")
        return data, elapsed

    print(f"\n[PRECOMPUTE START] Processing session {year} {race}...")
    t0 = time.time()
    
    session = fastf1.get_session(year, race, "R")
    session.load(telemetry=False, weather=True)
    
    total_laps = session.laps['LapNumber'].max()
    if pd.isna(total_laps) or total_laps is None:
        total_laps = 60

    total_laps = int(total_laps)
    print(f"Loaded {race} ({total_laps} laps). Computing field_state across all laps...")

    field_state = {}

    for lap in range(1, total_laps + 1):
        # 1. Phase 1: Tire Degradation Engine (Field-wide)
        field_deg = get_field_degradation(session, lap)

        # 2. Phase 2: Safety Car / VSC Probability Agent
        sc_prob = compute_sc_vsc_probability(session, lap)

        # 3. Phase 4: Weather Nowcast Agent
        weather = compute_weather_nowcast(session, lap)

        # 4. Phase 3: Rival Strategy Inference Engine (every driver at lap L)
        rival_map = {}
        if field_deg:
            for d_info in field_deg:
                d_code = d_info['driver']
                rival_res = infer_rival_strategy(session, lap, d_code, precomputed_field_deg=field_deg)
                if rival_res:
                    rival_map[d_code] = rival_res

        field_state[str(lap)] = {
            "lap_number": lap,
            "total_laps": total_laps,
            "sc_probability": sc_prob,
            "weather_nowcast": weather,
            "field_degradation": field_deg,
            "rival_inference": rival_map
        }

    t1 = time.time()
    execution_time = t1 - t0
    print(f"[PRECOMPUTE COMPLETE] Processed {total_laps} laps in {execution_time:.2f}s.")

    # Sanitize and write JSON to cache
    clean_field_state = sanitize_for_json(field_state)
    
    with open(filepath, "w") as f:
        json.dump(clean_field_state, f, indent=2)

    file_size_kb = os.path.getsize(filepath) / 1024.0
    print(f"[CACHE SAVED] Cached field_state saved to: {filepath} ({file_size_kb:.1f} KB)")

    return clean_field_state, execution_time

def load_precomputed_field_state(year, race):
    """
    Helper to instantly load cached field_state from disk without recomputing.
    """
    filepath = get_cache_filepath(year, race)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No precomputed cache found for {year} {race} at {filepath}. Run precompute first.")
    
    with open(filepath, "r") as f:
        return json.load(f)

def run_tests():
    print("Running self-contained verification tests for Precompute Pipeline (Phase 5)...")
    
    # Precompute 2023 Dutch Grand Prix
    fs, exec_time = precompute_race_field_state(2023, "Dutch Grand Prix", force_recompute=True)
    assert fs is not None, "Field state should not be None"
    assert "1" in fs and "40" in fs, "Field state must contain lap keys"
    
    # Check structure at Lap 40
    lap40 = fs["40"]
    assert "sc_probability" in lap40, "Lap 40 must contain sc_probability"
    assert "weather_nowcast" in lap40, "Lap 40 must contain weather_nowcast"
    assert "field_degradation" in lap40, "Lap 40 must contain field_degradation"
    assert "rival_inference" in lap40, "Lap 40 must contain rival_inference"
    
    # Test instant reload
    t0 = time.time()
    reloaded_fs = load_precomputed_field_state(2023, "Dutch Grand Prix")
    reload_time_ms = (time.time() - t0) * 1000.0
    assert reloaded_fs is not None, "Reloaded field state should not be None"
    assert reload_time_ms < 500.0, f"Cache reload should be sub-second, took {reload_time_ms:.2f}ms"
    print(f"[PASS] Precompute pipeline unit verification successful. Disk reload time: {reload_time_ms:.2f}ms")

    print("\nALL PHASE 5 PRECOMPUTE PIPELINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Pipeline (Phase 5)")
    parser.add_argument("--year", type=int, default=2023, help="Race year")
    parser.add_argument("--race", type=str, default="Dutch Grand Prix", help="Race name")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if cache exists")
    parser.add_argument("--test", action="store_true", help="Run self-contained verification tests")

    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    fs, exec_time = precompute_race_field_state(args.year, args.race, force_recompute=args.force)

    print(f"\n### Precompute Summary — {args.year} {args.race} ###\n")
    print(f"- **Total Laps Precomputed**: {len(fs)}")
    print(f"- **Precompute Execution Time**: {exec_time:.2f}s")
    print(f"- **Cache Location**: {get_cache_filepath(args.year, args.race)}")
