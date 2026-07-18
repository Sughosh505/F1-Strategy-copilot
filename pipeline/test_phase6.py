import json
import sys
import os

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.orchestrator import evaluate_snapshot

def run_tests():
    print("=== Phase 6: Orchestrator Constraint Tests ===\n")

    # 1. High pit window, low SC prob
    snap1 = {
        "lap": 45,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "VER",
            "pit_urgency": "high",
            "degradation_score": 0.92,
            "pace_delta_vs_fresh": 1.5
        },
        "sc_probability": {
            "next_3_laps": 0.05,
            "next_5_laps": 0.08,
            "basis": "Low historical crash rate."
        },
        "field_context": []
    }
    
    print("\n--- Test 1: High pit urgency, low SC probability ---")
    print("Input Snapshot:")
    print(json.dumps(snap1, indent=2))
    res1 = evaluate_snapshot(snap1)
    print("Output Recommendation:")
    print(json.dumps(res1, indent=2))
    assert res1["call"] == "pit_now"


    # 2. Fresh tires, high SC prob
    snap2 = {
        "lap": 15,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "HAM",
            "pit_urgency": "low",
            "degradation_score": 0.15,
            "pace_delta_vs_fresh": 0.2
        },
        "sc_probability": {
            "next_3_laps": 0.85,
            "next_5_laps": 0.95,
            "basis": "Historical SC rate is high here."
        },
        "field_context": []
    }

    print("\n--- Test 2: Fresh tires (low urgency), HIGH SC probability ---")
    print("Input Snapshot:")
    print(json.dumps(snap2, indent=2))
    res2 = evaluate_snapshot(snap2)
    print("Output Recommendation:")
    print(json.dumps(res2, indent=2))
    assert res2["call"] != "pit_now"  # Should NOT pit just for SC prob with fresh tires.


    # 3. Active SC/VSC
    snap3 = {
        "lap": 22,
        "track_status": "Safety Car Active",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "NOR",
            "pit_urgency": "medium",
            "degradation_score": 0.55,
            "pace_delta_vs_fresh": 0.8
        },
        "sc_probability": {
            "next_3_laps": 0.99,
            "next_5_laps": 0.99,
            "basis": "Safety Car is CURRENTLY DEPLOYED."
        },
        "field_context": []
    }

    print("\n--- Test 3: Active Safety Car ---")
    print("Input Snapshot:")
    print(json.dumps(snap3, indent=2))
    res3 = evaluate_snapshot(snap3)
    print("Output Recommendation:")
    print(json.dumps(res3, indent=2))
    assert res3["call"] == "pit_now"


    # 4. Boring lap
    snap4 = {
        "lap": 8,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "LEC",
            "pit_urgency": "low",
            "degradation_score": 0.20,
            "pace_delta_vs_fresh": 0.3
        },
        "sc_probability": {
            "next_3_laps": 0.10,
            "next_5_laps": 0.15,
            "basis": "Standard risk."
        },
        "field_context": []
    }

    print("\n--- Test 4: Boring lap (low urgency, low SC prob) ---")
    print("Input Snapshot:")
    print(json.dumps(snap4, indent=2))
    res4 = evaluate_snapshot(snap4)
    print("Output Recommendation:")
    print(json.dumps(res4, indent=2))
    assert res4["call"] == "stay_out"

    print("\n[PASS] All 4 orchestrator scenarios behaved as constrained.")

if __name__ == "__main__":
    run_tests()
