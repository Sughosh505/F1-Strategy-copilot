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

# Import Phase 1 degradation logic relationally
from agents.tire_degradation import compute_driver_degradation, get_field_degradation

# Load environment variables
load_dotenv()
cache_dir = os.getenv("CACHE_DIR", "cache")
fastf1.Cache.enable_cache(cache_dir)

# Compound lifespan priors (laps)
COMPOUND_LIFESPANS = {
    'SOFT': {'median_life': 18.0, 'window_start': 14.0, 'window_end': 22.0},
    'MEDIUM': {'median_life': 26.0, 'window_start': 20.0, 'window_end': 32.0},
    'HARD': {'median_life': 38.0, 'window_start': 30.0, 'window_end': 45.0},
    'INTERMEDIATE': {'median_life': 20.0, 'window_start': 14.0, 'window_end': 26.0},
    'WET': {'median_life': 16.0, 'window_start': 10.0, 'window_end': 22.0},
    'DEFAULT': {'median_life': 22.0, 'window_start': 16.0, 'window_end': 28.0}
}

def compute_pit_probability(compound, tyre_life, deg_score, pit_urgency):
    """
    Computes pit probability (0.0 to 1.0) based on tire life vs compound lifespan priors
    and Phase 1 degradation score/urgency.
    """
    compound_str = str(compound).upper()
    info = COMPOUND_LIFESPANS.get(compound_str, COMPOUND_LIFESPANS['DEFAULT'])
    median_life = info['median_life']

    life_ratio = tyre_life / median_life

    if life_ratio < 0.75:
        base_prob = 0.05 + 0.20 * (life_ratio / 0.75)
    elif life_ratio <= 1.10:
        base_prob = 0.30 + 0.50 * ((life_ratio - 0.75) / 0.35)
    else:
        base_prob = min(0.95, 0.80 + 0.15 * (life_ratio - 1.10))

    # Boost for Phase 1 urgency
    if pit_urgency == 'HIGH':
        base_prob += 0.15
    elif pit_urgency == 'MEDIUM':
        base_prob += 0.05

    return max(0.0, min(1.0, base_prob))

def get_field_positions_and_gaps(session, current_lap):
    """
    Derives track positions, cumulative session time, and gap-to-leader at current_lap.
    Strictly forward-only: uses session.laps up to current_lap.
    Returns dictionary mapping driver -> dict with position, cumulative_time, gap_to_leader.
    """
    laps_up_to_current = session.laps[session.laps['LapNumber'] <= current_lap]
    
    # Get the latest completed lap per driver up to current_lap
    latest_laps = (
        laps_up_to_current
        .sort_values('LapNumber')
        .groupby('Driver')
        .last()
        .reset_index()
    )

    # Filter drivers who are on or near current_lap (within 1 lap)
    active_laps = latest_laps[latest_laps['LapNumber'] >= current_lap - 1].copy()

    if active_laps.empty:
        return {}

    # Sort drivers by LapNumber descending, then Time ascending
    # (Higher lap number completed first, then earlier session time)
    active_laps['TimeSec'] = active_laps['Time'].apply(
        lambda t: t.total_seconds() if not pd.isna(t) else np.nan
    )
    active_laps = active_laps.sort_values(by=['LapNumber', 'TimeSec'], ascending=[False, True]).reset_index(drop=True)

    leader_time = active_laps.iloc[0]['TimeSec']
    
    position_map = {}
    for idx, row in active_laps.iterrows():
        driver = row['Driver']
        pos = idx + 1
        cum_time = row['TimeSec']
        gap_to_leader = cum_time - leader_time if not pd.isna(cum_time) and not pd.isna(leader_time) else 0.0
        position_map[driver] = {
            'position': pos,
            'lap_number': int(row['LapNumber']),
            'cum_time': cum_time,
            'gap_to_leader': gap_to_leader
        }

    return position_map

def infer_rival_strategy(session, current_lap, target_driver):
    """
    Computes rival strategy metrics and undercut threats for target_driver at current_lap.
    Reuses Phase 1 degradation logic relationally across the field.
    Filters the 20-car field down to a concise, high-relevance field_context.
    """
    # 1. Fetch Phase 1 degradation metrics field-wide
    field_deg = get_field_degradation(session, current_lap)
    if not field_deg:
        return None

    # Map driver -> deg dict
    deg_map = {d['driver']: d for d in field_deg}

    if target_driver.upper() not in deg_map:
        return None

    target_deg = deg_map[target_driver.upper()]

    # 2. Derive field positions and gaps
    pos_map = get_field_positions_and_gaps(session, current_lap)
    if target_driver.upper() not in pos_map:
        return None

    target_pos_info = pos_map[target_driver.upper()]
    target_pos = target_pos_info['position']
    target_cum_time = target_pos_info['cum_time']

    # 3. Process each rival driver
    rival_analysis = []
    
    for d_code, d_deg in deg_map.items():
        if d_code == target_driver.upper():
            continue

        if d_code not in pos_map:
            continue

        r_pos_info = pos_map[d_code]
        r_pos = r_pos_info['position']
        r_cum_time = r_pos_info['cum_time']

        # Gap relative to target driver (positive = rival is behind target)
        gap_to_target = r_cum_time - target_cum_time if (r_cum_time and target_cum_time) else 999.0
        
        # Calculate pit probability for rival
        pit_prob = compute_pit_probability(
            d_deg['compound'],
            d_deg['tyre_life'],
            d_deg['degradation_score'],
            d_deg['pit_urgency']
        )

        # Relational degradation & pace deltas vs target driver
        delta_deg = d_deg['degradation_score'] - target_deg['degradation_score']
        delta_pace = d_deg['pace_delta_vs_fresh'] - target_deg['pace_delta_vs_fresh']
        delta_age = d_deg['tyre_life'] - target_deg['tyre_life']

        # Determine undercut threat conditions:
        # 1. Rival is behind target (gap > 0)
        # 2. Rival is within immediate undercut gap range (0 < gap <= 4.5s)
        # 3. Rival pit window open (pit_prob >= 0.45 or pit_urgency in ['MEDIUM', 'HIGH'])
        # 4. Target vulnerable (target deg_score >= 0.40 or rival tires significantly fresher)
        is_behind = (gap_to_target > 0.0) and (gap_to_target <= 25.0)
        in_undercut_range = (gap_to_target > 0.0) and (gap_to_target <= 4.5)
        rival_pit_window_open = (pit_prob >= 0.45) or (d_deg['pit_urgency'] in ['MEDIUM', 'HIGH'])
        target_vulnerable = (target_deg['degradation_score'] >= 0.40) or (delta_deg < -0.15)

        is_undercut_threat = is_behind and in_undercut_range and rival_pit_window_open and target_vulnerable

        # Determine if target can undercut rival ahead
        # (Rival ahead within 4.5s, target has high pit urgency or rival vulnerable)
        is_ahead = (gap_to_target < 0.0) and (gap_to_target >= -25.0)
        in_overcut_undercut_target_range = (gap_to_target < 0.0) and (gap_to_target >= -4.5)
        can_undercut_rival = is_ahead and in_overcut_undercut_target_range and (target_deg['pit_urgency'] in ['MEDIUM', 'HIGH'])

        # 4. Relevance Filter (Field Context Selection)
        # Retain car ONLY IF it meets strict relevance criteria:
        is_immediate_battle = (abs(r_pos - target_pos) == 1) and (abs(gap_to_target) <= 12.0)
        is_traffic_rejoin_window = (gap_to_target >= 18.0) and (gap_to_target <= 24.0)

        is_relevant = is_immediate_battle or is_undercut_threat or can_undercut_rival or is_traffic_rejoin_window

        # Templated basis string per rival
        if is_undercut_threat:
            rival_basis = (
                f"{d_code} (P{r_pos}, +{gap_to_target:.1f}s behind) is an IMMEDIATE UNDERCUT THREAT! "
                f"Tire age: {d_deg['tyre_life']} laps ({d_deg['compound']}), Pit Prob: {pit_prob*100:.1f}%. "
                f"Target P{target_pos} ({target_driver}) deg score is {target_deg['degradation_score']:.2f}."
            )
        elif can_undercut_rival:
            rival_basis = (
                f"{d_code} (P{r_pos}, {abs(gap_to_target):.1f}s ahead) is VULNERABLE TO UNDERCUT by {target_driver}. "
                f"Tire age: {d_deg['tyre_life']} laps ({d_deg['compound']})."
            )
        elif is_traffic_rejoin_window:
            rival_basis = (
                f"{d_code} (P{r_pos}, +{gap_to_target:.1f}s behind) is in Target's PIT REJOIN TRAFFIC WINDOW (~21s pit loss)."
            )
        else:
            rival_basis = f"{d_code} (P{r_pos}, gap: {gap_to_target:+.1f}s, Pit Prob: {pit_prob*100:.1f}%)."

        rival_info = {
            'driver': d_code,
            'position': r_pos,
            'gap_to_target_sec': gap_to_target,
            'compound': d_deg['compound'],
            'tyre_life': d_deg['tyre_life'],
            'stint': d_deg['stint'],
            'degradation_score': d_deg['degradation_score'],
            'pit_urgency': d_deg['pit_urgency'],
            'pit_probability': pit_prob,
            'delta_deg_vs_target': delta_deg,
            'delta_pace_vs_target': delta_pace,
            'undercut_threat': is_undercut_threat,
            'can_be_undercut_by_target': can_undercut_rival,
            'is_relevant': is_relevant,
            'basis': rival_basis
        }

        rival_analysis.append(rival_info)

    # Filter for field_context (only relevant drivers)
    field_context = [r for r in rival_analysis if r['is_relevant']]
    # Sort field_context by gap to target ascending
    field_context = sorted(field_context, key=lambda x: x['gap_to_target_sec'])

    # Count total active undercut threats
    threat_count = sum(1 for r in field_context if r['undercut_threat'])

    # Templated overall basis string
    if threat_count > 0:
        threat_names = ", ".join([r['driver'] for r in field_context if r['undercut_threat']])
        summary_basis = (
            f"ALERT: {threat_count} active undercut threat(s) detected for {target_driver.upper()} "
            f"at Lap {current_lap} ({threat_names}). Relevance filter selected {len(field_context)}/20 cars."
        )
    else:
        summary_basis = (
            f"No immediate undercut threats detected for {target_driver.upper()} at Lap {current_lap}. "
            f"Relevance filter selected {len(field_context)}/20 cars in active strategic window."
        )

    return {
        'target_driver': target_driver.upper(),
        'target_position': target_pos,
        'current_lap': int(current_lap),
        'target_compound': target_deg['compound'],
        'target_tyre_life': target_deg['tyre_life'],
        'target_deg_score': target_deg['degradation_score'],
        'target_pit_urgency': target_deg['pit_urgency'],
        'threat_count': threat_count,
        'field_context_count': len(field_context),
        'total_field_count': len(deg_map),
        'field_context': field_context,
        'summary_basis': summary_basis
    }

def run_tests():
    print("Running self-contained verification tests for Rival Inference Engine (Phase 3)...")

    # 1. Pit probability formula unit test
    p_soft_fresh = compute_pit_probability('SOFT', 5, 0.1, 'LOW')
    assert p_soft_fresh < 0.25, f"Fresh SOFT should have low pit probability, got {p_soft_fresh}"

    p_soft_worn = compute_pit_probability('SOFT', 20, 0.7, 'HIGH')
    assert p_soft_worn > 0.70, f"Worn SOFT (age 20, HIGH urgency) should have high pit probability, got {p_soft_worn}"
    print("[PASS] Pit probability math formula functions correctly.")

    # 2. Integration test on 2023 Dutch Grand Prix
    print("Loading cached 2023 Dutch Grand Prix for E2E Rival Inference validation...")
    session = fastf1.get_session(2023, "Dutch Grand Prix", "R")
    session.load(telemetry=False, weather=False)

    # Test Lap 12 (Known pit window / undercut moment where Alonso/Verstappen/Perez strategies intersected)
    res_lap12 = infer_rival_strategy(session, 12, "PER")
    assert res_lap12 is not None, "Lap 12 result for PER should not be None"
    assert res_lap12['target_driver'] == "PER", "Target driver should be PER"
    assert res_lap12['field_context_count'] < res_lap12['total_field_count'], "Relevance filter must exclude distant cars"
    print(f"[PASS] E2E validation passed for PER at Lap 12. Filtered {res_lap12['field_context_count']}/{res_lap12['total_field_count']} cars. Threat count: {res_lap12['threat_count']}")

    # Test Lap 25 (Quiet lap with stable green-flag gaps)
    res_lap25 = infer_rival_strategy(session, 25, "VER")
    assert res_lap25 is not None, "Lap 25 result for VER should not be None"
    assert res_lap25['threat_count'] == 0, f"Lap 25 for VER should have 0 undercut threats (quiet lap), got {res_lap25['threat_count']}"
    print(f"[PASS] E2E validation passed for VER at Lap 25 (quiet lap). Threat count: {res_lap25['threat_count']}")

    print("\nALL PHASE 3 RIVAL INFERENCE ENGINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rival Strategy Inference Engine (Phase 3)")
    parser.add_argument("--year", type=int, default=2023, help="Race year")
    parser.add_argument("--race", type=str, default="Dutch Grand Prix", help="Race name")
    parser.add_argument("--lap", type=int, default=12, help="Lap number to evaluate")
    parser.add_argument("--driver", type=str, default="PER", help="Target driver code (e.g. PER, VER, ALO)")
    parser.add_argument("--test", action="store_true", help="Run self-contained verification tests")

    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    print(f"Loading session {args.year} {args.race}...")
    session = fastf1.get_session(args.year, args.race, "R")
    session.load(telemetry=False, weather=False)

    print(f"\nInferring rival strategy for driver {args.driver.upper()} at Lap {args.lap}...")
    res = infer_rival_strategy(session, args.lap, args.driver)

    if not res:
        print(f"Could not compute rival inference for driver {args.driver} at lap {args.lap}.")
        sys.exit(1)

    print(f"\n### Rival Strategy Inference Report — {res['target_driver']} Lap {res['current_lap']} (P{res['target_position']}) ###\n")
    print(f"**Target Compound/Age**: {res['target_compound']} ({res['target_tyre_life']} laps) | **Deg Score**: {res['target_deg_score']:.2f} ({res['target_pit_urgency']})")
    print(f"**Summary**: {res['summary_basis']}\n")

    print(f"#### Relevant Field Context ({res['field_context_count']}/{res['total_field_count']} drivers retained) ####\n")

    df_ctx = pd.DataFrame(res['field_context'])
    if not df_ctx.empty:
        cols_to_print = [
            'driver', 'position', 'gap_to_target_sec', 'compound', 'tyre_life', 
            'degradation_score', 'pit_probability', 'undercut_threat', 'can_be_undercut_by_target'
        ]
        print(df_ctx[cols_to_print].to_markdown(index=False))
        print("\n**Rival Details & Basis**:")
        for r in res['field_context']:
            print(f"- **{r['driver']}**: {r['basis']}")
    else:
        print("No relevant rival cars in immediate strategic window.")
