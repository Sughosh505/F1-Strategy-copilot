# FastF1 Data Schema & Quirks (Phase 0 Output)

This document details the structure of the data returned by the FastF1 library and notes specific quirks that must be handled by the rule-based strategy agents.

---

## 1. Development & Cache Setup
FastF1 telemetry and timing data can be extremely heavy. To prevent long download times and API rate limiting, local caching must be enabled:
```python
import fastf1
import os

cache_dir = os.getenv("CACHE_DIR", "cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
```
*Note: A cache folder named `cache/` has been created in the workspace root and is ignored by git.*

---

## 2. Tire Age & Pit Stop Quirks

### Stint Boundaries
- A pit stop is represented by a change in the `Stint` number (e.g., `Stint` 1.0 to 2.0).
- The pit-in lap has a valid timestamp in `PitInTime` (e.g., `0 days 01:05:25.519000`) and `PitOutTime` is `NaT`.
- The pit-out lap has a valid timestamp in `PitOutTime` (e.g., `0 days 01:05:45.581000`) and `PitInTime` is `NaT`.
- Pit stops always span across the end of the pit-in lap and the beginning of the pit-out lap.

### Tire Life Reset
- **Quirk:** `TyreLife` does **not** always start at `1.0` at the beginning of a new stint.
- If a driver fits a brand-new set of tires, `FreshTyre` is `True` and `TyreLife` starts at `1.0` (increasing by `1.0` each lap).
- If a driver fits a used/scrubbed set of tires (common in qualifying or chaotic wet/dry races), `FreshTyre` is `False` and `TyreLife` starts at its pre-existing cumulative wear value (e.g. `2.0`, `5.0`, or even `13.0` laps).
- Upstream agents must use `TyreLife` to measure actual tire wear, rather than counting the number of laps in the current stint.

### Pit Time Loss
- The time lost during a pit stop is split across the pit-in lap (pit lane entry) and the pit-out lap (pit lane exit and cold out-lap).
- To calculate the true strategy loss or compare lap-time pace, both the pit-in lap time and the pit-out lap time must be adjusted/ignored as out-of-family times.

---

## 3. Track Status Codes

### Schema
The track status history is available in `session.track_status` (which contains `Time`, `Status`, and a text `Message`).
Status codes are defined as:
- `1`: `AllClear` (Green flag / Track clear)
- `2`: `Yellow` (Sector-specific yellow flags)
- `3`: (Rare/Not used)
- `4`: `SCDeployed` (Safety Car)
- `5`: `Red` (Red Flag / Session suspended)
- `6`: `VSCDeployed` (Virtual Safety Car)
- `7`: `VSCEnding` (Virtual Safety Car ending / clearing)

### Lap-by-Lap Representation in `session.laps`
- **Quirk:** The `TrackStatus` column in the `session.laps` DataFrame is a **concatenated string** of status codes encountered at *any point* during that lap.
- For example, if a lap starts under green, encounters a yellow flag, and then a Safety Car is deployed, its `TrackStatus` is `'124'`.
- If a lap is entirely completed under Safety Car, its `TrackStatus` is `'4'`.
- Rule-based agents checking for incidents must use string containment checks (e.g. `'4' in str(lap['TrackStatus'])` to detect a Safety Car lap, or `'6' in str(lap['TrackStatus'])` for a VSC lap) rather than exact string equality.

---

## 4. Weather Nowcast Alignment

### Weather Data Schema
Weather data is retrieved using `session.weather_data` and contains:
`['Time', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']`

- `Time` is the session time of the sample.
- `Rainfall` is a Boolean (`True`/`False`) indicating whether it was raining at that time.

### Alignment Logic
- **Quirk:** Weather samples are recorded at roughly 1-minute intervals, which does not map 1-to-1 to lap completion times.
- To align weather data to a lap for a driver:
  1. Find weather samples where `Time` falls between the driver's lap start and end:
     `lap['LapStartTime'] <= weather_data['Time'] <= lap['Time']`
  2. If weather samples are found within this range, average the values (or check if any sample indicates `Rainfall = True`).
  3. If no weather samples are found (e.g., for very short laps), fall back to the closest weather sample prior to the lap start time:
     `weather_data[weather_data['Time'] < lap['LapStartTime']].tail(1)`

---

## 5. Candidate Demo Races
The following four races have been verified to load successfully with clean, complete data and represent highly strategic scenarios:

1. **2021 Russian Grand Prix** (Sochi): Late-race rain showers created a critical choice between staying out on dry slick tires or pitting for intermediates.
2. **2021 Abu Dhabi Grand Prix** (Yas Marina): A late Safety Car led to contrasting tire decisions between Hamilton and Verstappen.
3. **2023 Dutch Grand Prix** (Zandvoort): Extreme weather transitions (dry/wet/drying/torrential) forced multiple reactive pit calls and Safety Car restarts.
4. **2024 British Grand Prix** (Silverstone): Damp track, drying line, and subsequent late shower created overlapping tire-crossover windows.
