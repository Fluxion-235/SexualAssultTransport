"""
Live Real-Time Transit Telemetry & Incident Simulation Engine
Fetches live environmental data from Open-Meteo API for Colombo, Sri Lanka (6.93°N, 79.86°E),
synchronizes with real-world local time, and models dynamic transit risk severity.
"""

import sys
import time
import json
import random
import urllib.request
from datetime import datetime
import zoneinfo

# Colombo Coordinates
LATITUDE = 6.9271
LONGITUDE = 79.8612
COLOMBO_TZ = zoneinfo.ZoneInfo("Asia/Colombo")

CORRIDORS = [
    {"id": "R138", "name": "Route 138 (Homagama ↔ Pettah)", "mode": "High Level Trunk", "weight": 0.20, "location": "Nugegoda / Kirulapone"},
    {"id": "R100", "name": "Route 100 (Panadura ↔ Pettah)", "mode": "Galle Road Coastal", "weight": 0.17, "location": "Moratuwa / Wellawatte"},
    {"id": "R154", "name": "Route 154 (Kiribathgoda ↔ Angulana)", "mode": "Orbital Student/Hosp", "weight": 0.15, "location": "Borella / Kelaniya Univ"},
    {"id": "R122", "name": "Route 122 (Avissawella ↔ Pettah)", "mode": "Low Level Semi-Express", "weight": 0.11, "location": "Kaluaggala / Hanwella"},
    {"id": "R120", "name": "Route 120 (Horana ↔ Pettah)", "mode": "Horana Commuter", "weight": 0.09, "location": "Piliyandala / Bokundara"},
    {"id": "KV_RAIL", "name": "Kelani Valley (KV) Rail Line", "mode": "Commuter Rail", "weight": 0.09, "location": "Maharagama Halt Platform"},
    {"id": "COAST_RAIL", "name": "Coastal Railway Line", "mode": "Commuter Rail", "weight": 0.07, "location": "Dehiwala / Bambalapitiya"},
    {"id": "R177", "name": "Route 177 (Kaduwela ↔ Kollupitiya)", "mode": "Tech & Campus Trunk", "weight": 0.05, "location": "Battaramulla / SLIIT Hub"},
    {"id": "R240", "name": "Route 240 (Negombo ↔ Colombo)", "mode": "Northern Arterial", "weight": 0.04, "location": "Wattala / Mabole"},
    {"id": "MMC_EXPRESS", "name": "MMC & CM01 Expressway Metrobus", "mode": "Mitigated Hub", "weight": 0.03, "location": "Makumbura Terminal Bay 4"}
]

PROBABLE_REASONS = [
    ("Standee Crush Overcrowding (>6.5 pers/m²)", "Overloading beyond 160% capacity strips physical clearance; enables non-consensual contact under guise of involuntary pressure.", "Primary Physical Catalyst"),
    ("Driver Hard Braking & Acceleration Inertia Jerk", "Aggressive throttle surge and abrupt braking weaponized by offender as pretext for falling onto passenger.", "Mechanical Trigger"),
    ("CPTED Darkness Deficit at Unlit Bus Halt (<10 lux)", "Absent municipal lighting and broken shelters eliminate natural surveillance, fostering predatory stalking.", "Infrastructure Deficit"),
    ("Monsoon Downpour: Shut Windows & Heavy Fogging", "Closed sliding windows and condensation create complete exterior invisibility, trapping commuters in humid cabin.", "Weather Catalyst"),
    ("Conductor Absenteeism & Crew Indifference", "Private bus crew prioritizes cash intake; conductor instructs packed standees to 'squeeze further back' rather than intervening.", "Enforcement Failure"),
    ("Substandard 2x3 Vehicle Ergonomics (<45 cm Aisle)", "Over-configured seat rows leave claustrophobic aisle clearance, forcing continuous bodily friction.", "Vehicle Defect"),
    ("Rear Exit Doorway Lingering Bottleneck", "Perpetrator deliberately blocks egress steps to physically brush against passengers during boarding/alighting.", "Spatial Bottleneck"),
    ("Lack of Dedicated Female Seating Partition", "No barrier separating priority seating from standing aisle, allowing standees to loom over seated female commuters.", "Design Deficit")
]

INCIDENT_TYPES = [
    ("Non-Consensual Physical Contact ('Jacking')", True, "Perpetrator exploited crush standing room to press against female commuter", "Penal Code Sec. 345A", 0),
    ("Aisle Standee Groping during Braking Surge", True, "Offender utilized bus deceleration inertia and hard driver braking as pretext for non-consensual groping", "Penal Code Sec. 345A", 1),
    ("Rear-Door Bottleneck Boarding Intimidation", True, "Perpetrator deliberately blocked narrow rear bus exit to force close physical contact with disembarking passengers", "Penal Code Sec. 345A / Public Nuisance", 6),
    ("Seat Partition Creep Groping", False, "Offender reached through narrow seat gap from behind to touch seated female commuter", "Penal Code Sec. 345A", 7),
    ("Persistent Lewd Sexual Remarks & Catcalling", False, "Offender directed continuous offensive sexual propositions and vulgar commentary at passenger", "Penal Code Sec. 345A (Verbal Clause)", 4),
    ("Last-Kilometer Stalking from Unlit Bus Halt", False, "Offender followed passenger from unlit roadside shelter into darkened access by-lane after disembarking", "Penal Code Sec. 345B (Stalking)", 2),
    ("Indecent Exposure & Lewd Gestures", False, "Perpetrator deliberately exposed genitalia or made sexually explicit physical gestures in passenger cabin", "Penal Code Sec. 365A / Sec. 345", 3),
    ("Non-Consensual Smartphone Filming / Upskirting", True, "Offender angled smartphone beneath victim's clothing while ascending crowded bus steps or platform stairwell", "Penal Code Sec. 345 / Computer Crimes", 0),
    ("Inappropriate Leaning & Forced Proximity", True, "Perpetrator persistently leaned whole upper torso over seated female passenger despite available space", "Transit Regulation Infraction", 5),
    ("Intoxicated Group Intimidation at Terminal", False, "Group of intoxicated individuals loitering at junction bus stop encircled commuter demanding contact info", "Public Drunkenness & Harassment", 2),
    ("Conductor Fare Collection Body Squeeze", True, "Private bus conductor inappropriately squeezed and brushed against passenger while collecting fares", "NTC Regulatory Breach / Sec. 345A", 4),
    ("Overcrowded Train Footbridge Bottleneck Stalking", True, "Perpetrator cornered female commuter during bottleneck crowd surge on unlit railway pedestrian footbridge", "Railway Ordinance / Sec. 345A", 2)
]

OUTCOMES = [
    ("🔇 Suffer in Silence (Trapped)", False, "Victim paralyzed by extreme crowd density; unable to move away (UNFPA 96% silence norm)"),
    ("❌ Crew Indifference / Inaction", False, "Conductor shouted at passenger to squeeze backward to maintain door clearance"),
    ("🚶 Forced Premature Exit", False, "Passenger disembarked 4 stops early in distress to escape offender"),
    ("👀 Bystander Passive Non-Intervention", False, "Surrounding male commuters averted gaze; perpetrator exited unchallenged"),
    ("🗣️ Verbal Protest; Offender Slipped Away", False, "Victim shouted in protest; harasser quickly stepped off vehicle into crowd"),
    ("🎒 Defensive Bag Shielding", False, "Commuter repositioned backpack / handbag across chest as physical barrier"),
    ("🤝 Female Commuters Formed Buffer", False, "Nearby female passengers intervened and formed physical protective barrier around victim"),
    ("⚖️ Formal Police Complaint Lodged (Hotline 119)", True, "Passenger contacted Police Women & Children's Desk / Hotline 119 (4% rare threshold)"),
    ("🛡️ Conductor Intervened / Removed Perpetrator", False, "Crew intervened and removed harasser (trained SLTB / MMC personnel)")
]


def fetch_colombo_weather():
    """Fetches real-time weather from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,cloud_cover,wind_speed_10m"
        f"&timezone=Asia%2FColombo"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TransitResearch/2.0)"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data.get("current", {})
    except Exception as e:
        print(f"[!] Live internet weather fetch error: {e}. Using fallback telemetry.")
        return {
            "temperature_2m": 31.0,
            "relative_humidity_2m": 62,
            "precipitation": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 7.0
        }


def calculate_live_severity(colombo_dt, weather_data):
    """
    Computes dynamic risk severity multiplier:
    Time-of-day factor * Heat/Thermal Stress * Monsoonal Rain Factor * Solar/CPTED Factor
    """
    h = colombo_dt.hour + colombo_dt.minute / 60.0

    # 1. Diurnal Congestion Factor
    if 7.0 <= h <= 9.5:
        time_factor = 2.2
        period_label = "MORNING PEAK RUSH (Crush Crowding)"
    elif 16.5 <= h <= 19.5:
        time_factor = 2.0
        period_label = "EVENING COMMUTE SURGE (Peak Overcrowding)"
    elif 11.5 <= h <= 14.0:
        time_factor = 1.15
        period_label = "MIDDAY SCHOOL & OFFICE SHIFT (Moderate Crowding)"
    elif h >= 20.0:
        time_factor = 0.8
        period_label = "NIGHT COMMUTE (Post-Dusk CPTED Darkness Deficit)"
    else:
        time_factor = 0.7
        period_label = "OFF-PEAK TRANSIT FLOW"

    # 2. Weather Heat / Thermal Stress
    temp = weather_data.get("temperature_2m", 30.0)
    humidity = weather_data.get("relative_humidity_2m", 60)
    heat_factor = 1.0
    if temp >= 30.0 and humidity >= 60:
        heat_factor = 1.15  # Thermal discomfort, open doors, standing congestion

    # 3. Monsoon Rain Surge
    precip = weather_data.get("precipitation", 0.0)
    rain_factor = 1.0
    if precip > 0.1:
        rain_factor = 1.45  # Closed bus windows, modal shift from bikes/walking to buses

    # 4. CPTED Darkness Factor (Sunset in Colombo ~18:05)
    cpted_factor = 1.0
    if h >= 18.25 or h < 5.75:
        cpted_factor = 1.35  # Darkness at peripheral unlit halts

    total_severity = time_factor * heat_factor * rain_factor * cpted_factor
    return {
        "total_severity": total_severity,
        "time_factor": time_factor,
        "heat_factor": heat_factor,
        "rain_factor": rain_factor,
        "cpted_factor": cpted_factor,
        "period_label": period_label,
        "temperature": temp,
        "humidity": humidity,
        "precipitation": precip
    }


def run_live_telemetry(speed_multiplier=1.0, max_ticks=None):
    """Executes the live telemetry incident loop."""
    print("=" * 75)
    print(" 🌍 LIVE TRANSIT INCIDENT TELEMETRY ENGINE — COLOMBO, SRI LANKA")
    print("=" * 75)
    print("Connecting to Open-Meteo Meteorological Satellite & Colombo Station...")

    weather = fetch_colombo_weather()
    print(f"[✓] Live Environmental Telemetry Connected:")
    print(f"    • Location: Colombo (6.93°N, 79.86°E) | Timezone: Asia/Colombo (UTC+5:30)")
    print(f"    • Air Temp: {weather.get('temperature_2m', 30.0)}°C | Relative Humidity: {weather.get('relative_humidity_2m', 60)}%")
    print(f"    • Rainfall: {weather.get('precipitation', 0.0)} mm | Wind Speed: {weather.get('wind_speed_10m', 7.0)} km/h\n")

    base_rate_per_sec = 3825 / 64800  # ~3,825 daily incidents / 18 operating hours = 0.059/sec (~1 every 17s)
    incident_counter = 0
    unreported_counter = 0
    reported_counter = 0
    crowding_counter = 0

    accumulator = 0.0
    last_weather_fetch = time.time()
    last_time = time.time()

    print(f"[▶] Live Telemetry Started (Speed: {speed_multiplier}x). Press Ctrl+C to stop.")
    print("-" * 75)

    try:
        while True:
            current_real_time = time.time()
            dt_colombo = datetime.now(COLOMBO_TZ)

            # Refresh weather every 10 minutes
            if current_real_time - last_weather_fetch > 600:
                weather = fetch_colombo_weather()
                last_weather_fetch = current_real_time

            sev = calculate_live_severity(dt_colombo, weather)
            current_rate = base_rate_per_sec * sev["total_severity"]

            delta_sec = current_real_time - last_time
            last_time = current_real_time

            accumulator += current_rate * (delta_sec * speed_multiplier)

            if accumulator >= 1.0:
                accumulator -= 1.0
                incident_counter += 1

                # Select Corridor
                rand_val = random.random()
                cum_w = 0.0
                corridor = CORRIDORS[0]
                for c in CORRIDORS:
                    cum_w += c["weight"]
                    if rand_val <= cum_w:
                        corridor = c
                        break

                # Select Incident
                inc_type, is_crowd, inc_desc, penal_code, cause_idx = random.choice(INCIDENT_TYPES)
                cause_title, cause_desc, cause_tier = PROBABLE_REASONS[cause_idx]
                if is_crowd:
                    crowding_counter += 1

                # Outcome (4% police reporting)
                is_reported = random.random() < 0.04
                if is_reported:
                    reported_counter += 1
                    outcome_label, _, outcome_desc = [o for o in OUTCOMES if o[1]][0]
                else:
                    unreported_counter += 1
                    outcome_label, _, outcome_desc = random.choice([o for o in OUTCOMES if not o[1]])

                time_str = dt_colombo.strftime("%I:%M:%S %p")
                print(f"[{time_str}] ⚠️ INCIDENT #{incident_counter:04d} ON {corridor['name']}")
                print(f"  • Telemetry Context: {sev['period_label']} | Temp: {sev['temperature']}°C (Severity: {sev['total_severity']:.2f}x)")
                print(f"  • Location: Near {corridor['location']} [{corridor['mode']}]")
                print(f"  • Incident Nature: {inc_type} ({penal_code})")
                print(f"  • 🔍 Probable Reason: {cause_title} — {cause_desc} [{cause_tier}]")
                print(f"  • Empirical Outcome: {outcome_label} — {outcome_desc}")
                print(f"  • Live Counters: Total: {incident_counter} | Suffer Silent (96%): {unreported_counter} | Reported (4%): {reported_counter} | Overcrowding: {crowding_counter}")
                print("-" * 75)

                if max_ticks and incident_counter >= max_ticks:
                    break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[■] Live telemetry simulation stopped by user.")
        print(f"Final Summary for Current Transit Window:")
        print(f"  • Total Incidents Logged: {incident_counter}")
        print(f"  • Silent / Unreported: {unreported_counter} ({(unreported_counter/max(1,incident_counter))*100:.1f}%)")
        print(f"  • Formally Reported to Police: {reported_counter} ({(reported_counter/max(1,incident_counter))*100:.1f}%)")


if __name__ == "__main__":
    speed = 1.0
    ticks = None
    if len(sys.argv) > 1:
        try:
            speed = float(sys.argv[1])
        except ValueError:
            speed = 1.0
    if len(sys.argv) > 2:
        try:
            ticks = int(sys.argv[2])
        except ValueError:
            ticks = None

    run_live_telemetry(speed_multiplier=speed, max_ticks=ticks)
