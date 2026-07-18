import sys
import os

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.precompute import load_precomputed_field_state

def assemble_snapshot(year: int, race: str, lap: int, driver_code: str) -> dict:
    """
    Assembles a strict JSON snapshot for the orchestrator based on precomputed field_state.
    """
    field_state = load_precomputed_field_state(year, race)
    
    lap_key = str(lap)
    if lap_key not in field_state:
        raise ValueError(f"Lap {lap} not found in precomputed data for {year} {race}.")
    
    lap_data = field_state[lap_key]
    
    # 1. Triggered Car State
    triggered_car = None
    for car in lap_data.get("field_degradation", []):
        if car.get("driver") == driver_code:
            triggered_car = car
            break
            
    if not triggered_car:
        raise ValueError(f"Driver {driver_code} not found in field_degradation at lap {lap}.")
        
    # 2. Field Context (Rival Inference)
    rival_data = lap_data.get("rival_inference", {}).get(driver_code, {})
    field_context = rival_data.get("relevant_rivals", [])
    
    # 3. Track & Weather
    sc_prob = lap_data.get("sc_probability", {})
    weather = lap_data.get("weather_nowcast", {})
    
    # 4. Active Track Status (Override)
    is_sc = sc_prob.get("is_sc_active", False)
    is_vsc = sc_prob.get("is_vsc_active", False)
    
    if is_sc:
        track_status = "Safety Car Active"
    elif is_vsc:
        track_status = "Virtual Safety Car Active"
    else:
        track_status = "Green"
    
    snapshot = {
        "lap": lap,
        "track_status": track_status,
        "weather_trend": weather.get("trend_summary", "Unknown"),
        "triggered_car": {
            "driver": driver_code,
            "pit_urgency": triggered_car.get("pit_urgency", "low"),
            "degradation_score": triggered_car.get("degradation_score", 0.0),
            "pace_delta_vs_fresh": triggered_car.get("pace_delta_vs_fresh", 0.0),
            "lookahead_degradation_score_next_3_laps": triggered_car.get("lookahead_degradation_score_next_3_laps", triggered_car.get("degradation_score", 0.0)),
            "lookahead_degradation_score_next_5_laps": triggered_car.get("lookahead_degradation_score_next_5_laps", triggered_car.get("degradation_score", 0.0)),
            "lookahead_trend_next_3_laps": triggered_car.get("lookahead_trend_next_3_laps", "flat/stable")
        },
        "sc_probability": {
            "next_3_laps": sc_prob.get("sc_vsc_prob_next_3_laps", 0.0),
            "next_5_laps": sc_prob.get("sc_vsc_prob_next_5_laps", 0.0),
            "basis": sc_prob.get("basis", "")
        },
        "field_context": field_context
    }
    
    return snapshot

if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--race", type=str, default="Dutch Grand Prix")
    parser.add_argument("--lap", type=int, default=40)
    parser.add_argument("--driver", type=str, default="VER")
    args = parser.parse_args()
    
    try:
        snap = assemble_snapshot(args.year, args.race, args.lap, args.driver)
        print(json.dumps(snap, indent=2))
    except Exception as e:
        print(f"Error: {e}")
