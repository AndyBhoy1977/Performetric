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




# ---------------------------------------------------------------------------
# Multi-match: identity, loading, and cross-match aggregation
# ---------------------------------------------------------------------------

def extract_opponent(df: pd.DataFrame) -> str:
    """Pull the opponent out of drill titles like '2. First Half v Denmark'."""
    for title in df["Drill"].unique():
        m = re.search(r"\bv\.?\s+([A-Z][\w' \-]+)$", str(title).strip())
        if m:
            return m.group(1).strip()
    return ""


def match_label(df: pd.DataFrame, fallback: str = "") -> str:
    """A stable, human-readable key for one match file."""
    date = ""
    if "Date" in df and len(df):
        date = str(df["Date"].iloc[0]).strip()
    opp = extract_opponent(df)
    if date and opp:
        return f"{date} v {opp}"
    if date:
        return date
    return opp or fallback or "Unlabelled match"


def _sort_key(label: str):
    """Sort match labels chronologically on a leading dd/mm/yyyy where present."""
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", str(label))
    return (int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else (0, 0, 0)


def load_many(files) -> pd.DataFrame:
    """Load several StatSports exports into one frame with a Match column.

    `files` is a list of (name, file-like) pairs. Files that fail to parse are
    skipped and reported, so one bad export doesn't sink the whole library.
    """
    frames, errors = [], []
    for name, handle in files:
        try:
            df, _ = load_gps(handle)
        except Exception as exc:
            errors.append((name, str(exc)))
            continue
        df = df.copy()
        df["Match"] = match_label(df, fallback=str(name))
        df["Source file"] = str(name)
        frames.append(df)

    if not frames:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(frames, ignore_index=True)
        # De-duplicate: same match uploaded twice keeps the first copy.
        combined = combined.drop_duplicates(
            subset=["Match", "Player", "Drill"], keep="first"
        )
    combined.attrs["errors"] = errors
    return combined


def ordered_matches(df: pd.DataFrame) -> list:
    """Match labels, oldest first."""
    if df is None or df.empty or "Match" not in df:
        return []
    return sorted(df["Match"].unique(), key=_sort_key)


def match_period_rates(df: pd.DataFrame, min_minutes: float = 15) -> pd.DataFrame:
    """Exposure-weighted squad rates for every match x period in the library."""
    if df is None or df.empty:
        return pd.DataFrame()

    match_only = df[df["Period"].isin(MATCH_PERIODS)].copy()
    if match_only.empty:
        return pd.DataFrame()

    # Eligibility is judged per match, not across the library.
    exposure = match_only.groupby(["Match", "Player"])["Minutes"].sum().reset_index()
    keep = exposure[exposure["Minutes"] >= min_minutes][["Match", "Player"]]
    match_only = match_only.merge(keep, on=["Match", "Player"], how="inner")

    rows = []
    for (match, period), g in match_only.groupby(["Match", "Period"]):
        total_min = g["Minutes"].sum()
        if not total_min:
            continue
        row = {
            "Match": match,
            "Period": period,
            "Players": g["Player"].nunique(),
            "Total minutes": total_min,
            "Distance (m)": g["Distance (m)"].sum(),
            "m/min": g["Distance (m)"].sum() / total_min,
        }
        if "HSR (m)" in g:
            row["HSR (m)"] = g["HSR (m)"].sum()
            row["HSR/min"] = g["HSR (m)"].sum() / total_min
        if "Accels" in g and "Decels" in g:
            row["Mechanical load/min"] = (g["Accels"].sum() + g["Decels"].sum()) / total_min
        if "Max speed" in g:
            row["Peak speed"] = g["Max speed"].max()
        rows.append(row)

    out = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(MATCH_PERIODS)}
    return out.sort_values(
        ["Match", "Period"],
        key=lambda s: s.map(_sort_key) if s.name == "Match" else s.map(order),
    )


def match_totals(df: pd.DataFrame, min_minutes: float = 15) -> pd.DataFrame:
    """One row per match: whole-match squad rates, halves combined."""
    rates = match_period_rates(df, min_minutes)
    if rates.empty:
        return pd.DataFrame()

    rows = []
    for match, g in rates.groupby("Match"):
        total_min = g["Total minutes"].sum()
        row = {
            "Match": match,
            "Total minutes": total_min,
            "Distance (m)": g["Distance (m)"].sum(),
            "m/min": g["Distance (m)"].sum() / total_min,
        }
        if "HSR (m)" in g:
            row["HSR/min"] = g["HSR (m)"].sum() / total_min
        if "Mechanical load/min" in g:
            row["Mechanical load/min"] = float(
                np.average(g["Mechanical load/min"], weights=g["Total minutes"])
            )
        if "Peak speed" in g:
            row["Peak speed"] = g["Peak speed"].max()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Match", key=lambda s: s.map(_sort_key))


def player_match_matrix(df: pd.DataFrame, metric: str = "m/min",
                        min_minutes: float = 15) -> pd.DataFrame:
    """Players down the side, matches across the top, for one rate metric."""
    if df is None or df.empty:
        return pd.DataFrame()

    match_only = df[df["Period"].isin(MATCH_PERIODS)].copy()
    if match_only.empty:
        return pd.DataFrame()

    source = {"m/min": ("Distance (m)", None),
              "HSR/min": ("HSR (m)", None),
              "Mechanical load/min": ("Accels", "Decels")}.get(metric)

    rows = []
    for (match, player), g in match_only.groupby(["Match", "Player"]):
        minutes = g["Minutes"].sum()
        if minutes < min_minutes:
            continue
        if source is None:
            value = g[metric].mean() if metric in g else np.nan
        else:
            a, b = source
            if a not in g:
                continue
            total = g[a].sum() + (g[b].sum() if b and b in g else 0)
            value = total / minutes
        rows.append({"Player": player, "Match": match, metric: value, "Minutes": minutes})

    if not rows:
        return pd.DataFrame()

    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="Player", columns="Match", values=metric, aggfunc="mean")
    return wide.reindex(columns=sorted(wide.columns, key=_sort_key))


# A standard deviation computed from a handful of matches is unstable: with three
# prior matches a z-score of 18 is arithmetic, not a finding. Below this many
# baseline matches the z-score is withheld and only the % change is shown.
Z_MIN_MATCHES = 5


def player_baselines(df: pd.DataFrame, metric: str = "m/min",
                     min_minutes: float = 15, min_matches: int = 3) -> pd.DataFrame:
    """Compare each player's latest match against their own previous average.

    This is the comparison that matters individually — a squad average hides the
    player who is 20% down on his own normal. Needs a few matches to mean anything,
    hence min_matches.
    """
    wide = player_match_matrix(df, metric, min_minutes)
    if wide.empty or wide.shape[1] < 2:
        return pd.DataFrame()

    latest_col = wide.columns[-1]
    prior = wide.iloc[:, :-1]

    rows = []
    for player, row in wide.iterrows():
        history = prior.loc[player].dropna()
        latest = row[latest_col]
        if pd.isna(latest) or len(history) < min(min_matches - 1, 1):
            continue
        mean = history.mean()
        sd = history.std(ddof=0)
        n = int(len(history))
        z = (latest - mean) / sd if sd and sd > 0 and n >= Z_MIN_MATCHES else np.nan
        rows.append({
            "Player": player,
            "Latest": latest,
            "Own average": mean,
            "Matches in baseline": n,
            "Change %": (latest - mean) / mean * 100 if mean else np.nan,
            "Z score": z,
        })

    out = pd.DataFrame(rows)
    return out.sort_values("Change %") if not out.empty else out
