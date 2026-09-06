"""
pipeline.py — StatSports export parsing, period classification, aggregation and QA.

Pure pandas. No Streamlit. This is the part you reuse across clients and test.
"""

import re
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Period classification — the heart of the pipeline
# ---------------------------------------------------------------------------

WHOLE_SESSION = "Whole session"

PERIOD_RULES = [
    (r"entire\s*session|whole\s*session|full\s*session", WHOLE_SESSION),
    (r"warm\s*up|warmup", "Warm up"),
    (r"first\s*half|1st\s*half|\bh1\b", "First half"),
    (r"second\s*half|2nd\s*half|\bh2\b", "Second half"),
    (r"extra\s*time|\bet\b", "Extra time"),
    (r"conditioning|post\s*match|extras|top\s*up", "Post-match"),
]

MATCH_PERIODS = ["First half", "Second half", "Extra time"]


def classify_period(title: str) -> str:
    """Map a free-text drill title to a canonical period label."""
    t = re.sub(r"^\s*\d+[\.\)]\s*", "", str(title)).strip().lower()
    for pattern, label in PERIOD_RULES:
        if re.search(pattern, t):
            return label
    return "Other"


def parse_duration(value) -> float:
    """HH:MM:SS or MM:SS -> minutes."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return np.nan
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    else:
        return float(parts[0])
    return h * 60 + m + sec / 60


# ---------------------------------------------------------------------------
# Column mapping — tolerant to StatSports naming variants
# ---------------------------------------------------------------------------

CANONICAL = {
    "player": ["player display name", "player name", "player", "athlete"],
    "date": ["date", "session date"],
    "drill": ["drill title", "period name", "drill", "split name"],
    "position": ["player primary position", "position", "primary position"],
    "duration": ["total time", "duration", "field time"],
    "distance": ["total distance", "distance (m)", "distance"],
    "dist_per_min": ["distance per min", "distance per minute", "m/min"],
    "hsr": ["high speed running (absolute)", "high speed running", "hsr"],
    "hsr_per_min": ["hsr per minute (absolute)", "hsr per min", "hsr per minute"],
    "sprints": ["sprints", "sprint count", "number of sprints"],
    "sprint_dist": ["sprint distance", "sprint distance (absolute)"],
    "max_speed": ["max speed", "maximum speed", "peak speed"],
    "accels": ["accelerations (absolute)", "accelerations", "accel count"],
    "decels": ["decelerations (absolute)", "decelerations", "decel count"],
    "accel_per_min": ["accelerations per min (absolute)", "accelerations per min"],
    "decel_per_min": ["decels per min (absolute)", "decelerations per min"],
    "explosive": ["explosive distance (absolute)", "explosive distance"],
    "metabolic_dist": ["equivalent metabolic distance", "metabolic distance"],
    "metabolic_power": ["average metabolic power", "metabolic power"],
    "speed_intensity": ["speed intensity"],
    "zone5": ["distance zone 5 (absolute)", "distance zone 5"],
    "zone6": ["distance zone 6 (absolute)", "distance zone 6"],
    "hr_max": ["max heart rate", "maximum heart rate"],
    "hr_avg": ["average heart rate", "avg heart rate", "mean heart rate"],
}


def map_columns(df: pd.DataFrame) -> dict:
    lookup = {c.strip().lower(): c for c in df.columns}
    found = {}
    for key, candidates in CANONICAL.items():
        for cand in candidates:
            if cand in lookup:
                found[key] = lookup[cand]
                break
    return found


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------


def load_gps(file) -> tuple:
    df = pd.read_csv(file)
    cols = map_columns(df)

    missing = [k for k in ("player", "drill", "distance") if k not in cols]
    if missing:
        raise ValueError(
            "Couldn't find required columns: "
            + ", ".join(missing)
            + ". Check the export includes player name, drill title and total distance."
        )

    out = pd.DataFrame()
    out["Player"] = df[cols["player"]].astype(str).str.strip()
    out["Drill"] = df[cols["drill"]].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["Period"] = out["Drill"].map(classify_period)
    out["Position"] = df[cols["position"]].astype(str).str.strip() if "position" in cols else "Unknown"
    out["Date"] = df[cols["date"]] if "date" in cols else ""

    out["Minutes"] = df[cols["duration"]].map(parse_duration) if "duration" in cols else np.nan

    numeric_keys = [
        "distance", "dist_per_min", "hsr", "hsr_per_min", "sprints", "sprint_dist",
        "max_speed", "accels", "decels", "accel_per_min", "decel_per_min",
        "explosive", "metabolic_dist", "metabolic_power", "speed_intensity",
        "zone5", "zone6", "hr_max", "hr_avg",
    ]
    labels = {
        "distance": "Distance (m)", "dist_per_min": "m/min", "hsr": "HSR (m)",
        "hsr_per_min": "HSR/min", "sprints": "Sprints", "sprint_dist": "Sprint distance (m)",
        "max_speed": "Max speed", "accels": "Accels", "decels": "Decels",
        "accel_per_min": "Accels/min", "decel_per_min": "Decels/min",
        "explosive": "Explosive distance (m)", "metabolic_dist": "Metabolic distance (m)",
        "metabolic_power": "Metabolic power", "speed_intensity": "Speed intensity",
        "zone5": "Zone 5 (m)", "zone6": "Zone 6 (m)", "hr_max": "Max HR", "hr_avg": "Avg HR",
    }
    for key in numeric_keys:
        if key in cols:
            out[labels[key]] = pd.to_numeric(df[cols[key]], errors="coerce")

    # Derive per-minute rates where the export didn't supply them
    if "m/min" not in out and "Distance (m)" in out:
        out["m/min"] = out["Distance (m)"] / out["Minutes"]
    if "Accels/min" not in out and "Accels" in out:
        out["Accels/min"] = out["Accels"] / out["Minutes"]
    if "Decels/min" not in out and "Decels" in out:
        out["Decels/min"] = out["Decels"] / out["Minutes"]

    # Mechanical load: accel + decel events per minute
    if "Accels/min" in out and "Decels/min" in out:
        out["Mechanical load/min"] = out["Accels/min"] + out["Decels/min"]

    return out, cols


def reconcile(df: pd.DataFrame) -> pd.DataFrame:
    """Compare whole-session totals against the sum of the individual drills."""
    rows = []
    for player, g in df.groupby("Player"):
        whole = g[g["Period"] == WHOLE_SESSION]["Distance (m)"].sum()
        parts = g[g["Period"] != WHOLE_SESSION]["Distance (m)"].sum()
        if whole > 0:
            rows.append(
                {
                    "Player": player,
                    "Whole session (m)": whole,
                    "Sum of drills (m)": parts,
                    "Unallocated (m)": whole - parts,
                    "Unallocated %": (whole - parts) / whole * 100,
                }
            )
    return pd.DataFrame(rows)


def quality_flags(df: pd.DataFrame) -> list:
    flags = []

    # Heart rate sanity
    if "Max HR" in df:
        hr = df["Max HR"].dropna()
        if len(hr) and hr.max() < 120:
            flags.append(
                ("bad", f"Heart rate is not usable. Highest max HR in the file is {hr.max():.0f} bpm — "
                        "below any plausible match value. Treat all HR-derived metrics as missing until "
                        "the belt/monitor pairing is checked.")
            )
    if "Avg HR" in df:
        low = df[(df["Avg HR"].notna()) & (df["Avg HR"] < 60)]
        if len(low):
            names = ", ".join(sorted(low["Player"].unique()))
            flags.append(("warn", f"Implausibly low average HR on some files: {names}. Likely dropout mid-session."))

    # Exposure
    match = df[df["Period"].isin(MATCH_PERIODS)]
    if len(match):
        by_player = match.groupby("Player")["Minutes"].sum()
        short = by_player[by_player < 20]
        if len(short):
            names = ", ".join(f"{p} ({m:.0f} min)" for p, m in short.items())
            flags.append(
                ("warn", f"Low match exposure — compare these on per-minute rates only, never totals: {names}.")
            )

    # Missing periods
    have_h1 = set(df[df["Period"] == "First half"]["Player"])
    have_h2 = set(df[df["Period"] == "Second half"]["Player"])
    no_match = set(df["Player"]) - have_h1 - have_h2
    if no_match:
        flags.append(("warn", f"No match period recorded (unused subs, or drill not split): {', '.join(sorted(no_match))}."))

    # Reconciliation
    rec = reconcile(df)
    if len(rec):
        big = rec[rec["Unallocated %"] > 25]
        if len(big):
            names = ", ".join(f"{r.Player} ({r._5:.0f}%)" for r in big.itertuples())
            flags.append(
                ("warn", f"More than a quarter of session distance sits outside the tagged drills for: {names}. "
                         "Usually half-time, warm-down or untagged rehab work.")
            )

    if not flags:
        flags.append(("ok", "No data quality issues detected in this file."))
    return flags


