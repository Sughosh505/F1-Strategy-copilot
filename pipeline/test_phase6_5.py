import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.orchestrator import evaluate_snapshot

def run_tests():
    print("=== Phase 6.5: Orchestrator Lookahead & Comparative Tests ===\n")

    # 1. High pit window, low SC prob, flat trend
    snap1 = {
        "lap": 45,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "VER",
            "pit_urgency": "high",
            "degradation_score": 0.92,
            "pace_delta_vs_fresh": 1.5,
            "lookahead_degradation_score_next_3_laps": 0.95,
            "lookahead_trend_next_3_laps": "flat/stable"
        },
        "sc_probability": {
            "next_3_laps": 0.05,
            "next_5_laps": 0.08,
            "basis": "Low historical crash rate."
        },
        "field_context": []
    }
    
    print("\n--- Test 1: High pit urgency, low SC probability, flat trend ---")
    print("Input Snapshot:")
    print(json.dumps(snap1, indent=2))
    res1 = evaluate_snapshot(snap1)
    print("Output Recommendation:")
    print(json.dumps(res1, indent=2))
    assert res1["call"] == "pit_now"


    # 2. Fresh tires, high SC prob, flat trend
    snap2 = {
        "lap": 15,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "HAM",
            "pit_urgency": "low",
            "degradation_score": 0.15,
            "pace_delta_vs_fresh": 0.2,
            "lookahead_degradation_score_next_3_laps": 0.18,
            "lookahead_trend_next_3_laps": "flat/stable"
        },
        "sc_probability": {
            "next_3_laps": 0.85,
            "next_5_laps": 0.95,
            "basis": "Historical SC rate is high here."
        },
        "field_context": []
    }

    print("\n--- Test 2: Fresh tires (low urgency), HIGH SC probability, flat trend ---")
    print("Input Snapshot:")
    print(json.dumps(snap2, indent=2))
    res2 = evaluate_snapshot(snap2)
    print("Output Recommendation:")
    print(json.dumps(res2, indent=2))
    assert res2["call"] != "pit_now"


    # 3. Active Safety Car
    snap3 = {
        "lap": 22,
        "track_status": "Safety Car Active",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "NOR",
            "pit_urgency": "medium",
            "degradation_score": 0.55,
            "pace_delta_vs_fresh": 0.8,
            "lookahead_degradation_score_next_3_laps": 0.65,
            "lookahead_trend_next_3_laps": "accelerating_degradation"
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


    # 4. Boring lap (low urgency, low SC prob, flat trend)
    snap4 = {
        "lap": 8,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "LEC",
            "pit_urgency": "low",
            "degradation_score": 0.2,
            "pace_delta_vs_fresh": 0.3,
            "lookahead_degradation_score_next_3_laps": 0.23,
            "lookahead_trend_next_3_laps": "flat/stable"
        },
        "sc_probability": {
            "next_3_laps": 0.1,
            "next_5_laps": 0.15,
            "basis": "Standard risk."
        },
        "field_context": []
    }

    print("\n--- Test 4: Boring lap (low urgency, low SC prob, flat trend) ---")
    print("Input Snapshot:")
    print(json.dumps(snap4, indent=2))
    res4 = evaluate_snapshot(snap4)
    print("Output Recommendation:")
    print(json.dumps(res4, indent=2))
    assert res4["call"] == "stay_out"

    # 5. Accelerating Trend Test (medium urgency crossing into high)
    snap5 = {
        "lap": 30,
        "track_status": "Green",
        "weather_trend": "Dry",
        "triggered_car": {
            "driver": "RUS",
            "pit_urgency": "medium",
            "degradation_score": 0.65,
            "pace_delta_vs_fresh": 1.1,
            "lookahead_degradation_score_next_3_laps": 0.85,
            "lookahead_trend_next_3_laps": "crossing_into_high_degradation"
        },
        "sc_probability": {
            "next_3_laps": 0.2,
            "next_5_laps": 0.2,
            "basis": "Standard risk."
        },
        "field_context": []
    }

    print("\n--- Test 5: Accelerating Trend (decision window verification) ---")
    print("Input Snapshot:")
    print(json.dumps(snap5, indent=2))
    res5 = evaluate_snapshot(snap5)
    print("Output Recommendation:")
    print(json.dumps(res5, indent=2))

    print("\n[PASS] All orchestrator scenarios generated successfully.")

if __name__ == "__main__":
    run_tests()
