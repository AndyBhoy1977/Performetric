"""
Match Context — a single-file Streamlit app.

Everything lives here on purpose. Split across modules this deployed fine locally
and kept failing on Streamlit Cloud whenever one file reached the repo and another
didn't. One file cannot go out of sync with itself.

Sections, in order:
  1. Config and palette
  2. Pipeline      — StatSports parsing, period classification, aggregation, QA
  3. PDF report    — ReportLab, prefixed RP_
  4. Pitch views   — Plotly pitch graphics, prefixed PC_
  5. Open data     — StatsBomb and Metrica loaders
  6. The app
"""

import io
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph



# ==========================================================================
# 2. PIPELINE — StatSports parsing and aggregation
# ==========================================================================

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

# ==========================================================================
# 3. PDF REPORT
# ==========================================================================

# Print palette — light, not the app's dark theme. Reports get printed.
RP_INK = colors.HexColor("#14191D")
BODY = colors.HexColor("#3C4A52")
RP_MUTED = colors.HexColor("#7C8C96")
RULE = colors.HexColor("#D6DEE2")
RP_GRASS = colors.HexColor("#2E9E5B")
RP_AMBER = colors.HexColor("#C4801E")
RP_CORAL = colors.HexColor("#C2503F")
WASH = colors.HexColor("#F2F6F7")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

BODY_STYLE = ParagraphStyle(
    "body", fontName="Helvetica", fontSize=9, leading=12.5, textColor=BODY,
)
READING_STYLE = ParagraphStyle(
    "reading", fontName="Helvetica", fontSize=9, leading=12.5, textColor=RP_INK,
    leftIndent=6,
)


SHORT = {
    "Total minutes": "Minutes",
    "Distance (m)": "Distance",
    "Mechanical load/min": "Mech/min",
    "In-possession m/min": "In poss",
    "Out-of-possession m/min": "Out poss",
    "HSR per unit of press": "HSR/press",
    "Mechanical cost of press": "Mech/press",
    "xG per 1000 m": "xG/1000m",
}


def _fmt(v, dp=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:,.{dp}f}"


def _delta(a, b):
    if not a or pd.isna(a) or pd.isna(b):
        return None
    return (b - a) / a * 100


BOTTOM = MARGIN + 26  # leave room for the footer rule and strapline


class _Report:
    def __init__(self, c: rl_canvas.Canvas, meta=None):
        self.c = c
        self.meta = meta or {}
        self.y = PAGE_H - MARGIN
        self.page = 1

    def ensure(self, needed):
        """Break to a new page when the next block won't fit."""
        if self.y - needed < BOTTOM:
            _footer(self.c, self.meta, self.page)
            self.c.showPage()
            self.page += 1
            self.y = PAGE_H - MARGIN
            return True
        return False

    def space(self, h):
        self.y -= h

    def rule(self, colour=RULE, width=0.6):
        self.c.setStrokeColor(colour)
        self.c.setLineWidth(width)
        self.c.line(MARGIN, self.y, PAGE_W - MARGIN, self.y)
        self.y -= 2

    def heading(self, text, size=10.5):
        self.ensure(size + 30)
        self.c.setFillColor(RP_INK)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= size + 4

    def caption(self, text, size=7.6):
        self.c.setFillColor(RP_MUTED)
        self.c.setFont("Helvetica", size)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= size + 5

    def paragraph(self, text, style=BODY_STYLE, indent=0):
        p = Paragraph(text, style)
        w = PAGE_W - 2 * MARGIN - indent
        _, h = p.wrap(w, self.y)
        self.ensure(h + 6)
        _, h = p.wrap(w, self.y)
        p.drawOn(self.c, MARGIN + indent, self.y - h)
        self.y -= h + 4


def _header(r: _Report, meta: dict):
    c = r.c
    c.setFillColor(RP_INK)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(MARGIN, r.y, meta.get("title", "Match report"))

    c.setFillColor(RP_GRASS)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(PAGE_W - MARGIN, r.y + 2, "MATCH CONTEXT")
    r.y -= 15

    c.setFillColor(RP_MUTED)
    c.setFont("Helvetica", 8.5)
    bits = [b for b in (meta.get("date"), meta.get("squad"), meta.get("source")) if b]
    c.drawString(MARGIN, r.y, "   |   ".join(bits))
    r.y -= 9
    r.rule(RP_INK, 1.1)
    r.space(12)


def _metric_strip(r: _Report, summary: pd.DataFrame, joined):
    """Headline rate metrics per period, side by side."""
    c = r.c
    metrics = [m for m in ("m/min", "HSR/min", "Mechanical load/min") if m in summary]
    if joined is not None and "PPDA" in joined:
        metrics.append("PPDA")

    src = joined if joined is not None else summary
    periods = list(src["Period"])
    if len(periods) < 1:
        return

    box_h = 46
    x0 = MARGIN
    usable = PAGE_W - 2 * MARGIN
    col_w = usable / (len(metrics) or 1)

    c.setFillColor(WASH)
    c.rect(x0, r.y - box_h, usable, box_h, stroke=0, fill=1)

    for i, metric in enumerate(metrics):
        cx = x0 + i * col_w + 8
        c.setFillColor(RP_MUTED)
        c.setFont("Helvetica", 7)
        c.drawString(cx, r.y - 12, metric.upper())

        vals = []
        for p in periods:
            row = src[src["Period"] == p]
            vals.append(float(row[metric].iloc[0]) if len(row) and metric in row else float("nan"))

        c.setFillColor(RP_INK)
        c.setFont("Helvetica-Bold", 13)
        txt = "  ".join(_fmt(v, 1 if metric != "PPDA" else 1) for v in vals)
        c.drawString(cx, r.y - 28, txt)

        if len(vals) >= 2:
            d = _delta(vals[0], vals[1])
            if d is not None:
                # For PPDA a fall means a more aggressive press, so green it.
                good_down = metric == "PPDA"
                colour = RP_GRASS if ((d < 0) == good_down) else RP_AMBER
                c.setFillColor(colour)
                c.setFont("Helvetica-Bold", 8)
                c.drawString(cx, r.y - 39, f"{d:+.1f}% H1 to H2")

        if i:
            c.setStrokeColor(RULE)
            c.setLineWidth(0.5)
            c.line(x0 + i * col_w, r.y - box_h + 6, x0 + i * col_w, r.y - 6)

    r.y -= box_h + 14


def _period_table(r: _Report, table: pd.DataFrame, columns):
    c = r.c
    cols = [col for col in columns if col in table.columns]
    if not cols:
        return

    r.ensure(30 + 13 * len(table))
    usable = PAGE_W - 2 * MARGIN
    first_w = 62
    other_w = (usable - first_w) / max(len(cols) - 1, 1)

    def xpos(i):
        return MARGIN + (0 if i == 0 else first_w + (i - 1) * other_w)

    c.setFillColor(RP_MUTED)
    c.setFont("Helvetica-Bold", 7)
    for i, col in enumerate(cols):
        label = SHORT.get(col, col)
        if i == 0:
            c.drawString(xpos(i), r.y, label.upper())
        else:
            c.drawRightString(xpos(i) + other_w - 6, r.y, label.upper())
    r.y -= 6
    r.rule()
    r.space(9)

    for _, row in table.iterrows():
        c.setFillColor(RP_INK)
        for i, col in enumerate(cols):
            v = row[col]
            if i == 0:
                c.setFont("Helvetica-Bold", 8.5)
                c.drawString(xpos(i), r.y, str(v))
            else:
                c.setFont("Helvetica", 8.5)
                dp = 2 if isinstance(v, float) and abs(v) < 20 else 1
                txt = _fmt(v, dp) if isinstance(v, (int, float)) else str(v)
                c.drawRightString(xpos(i) + other_w - 6, r.y, txt)
        r.y -= 13

    r.space(2)
    r.rule()
    r.space(12)


def _panel(d, x, y, w, h, title, labels, values, colour, fmt="{:.1f}"):
    """One small bar panel with its own scale. Clearer in print than a dual axis."""
    d.add(String(x, y + h + 12, title, fontName="Helvetica-Bold", fontSize=7.5, fillColor=colour))

    vmax = max(values) * 1.30 if max(values) else 1.0
    n = len(values)
    slot = w / n
    bw = min(30, slot * 0.46)

    d.add(Line(x, y, x + w, y, strokeColor=RULE, strokeWidth=0.7))

    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = x + slot * (i + 0.5)
        bh = (v / vmax) * h
        d.add(Rect(cx - bw / 2, y, bw, bh, fillColor=colour, strokeColor=None,
                   fillOpacity=0.22))
        d.add(Line(cx - bw / 2, y + bh, cx + bw / 2, y + bh, strokeColor=colour, strokeWidth=2))
        d.add(String(cx, y + bh + 4, fmt.format(v), fontName="Helvetica-Bold", fontSize=8.5,
                     fillColor=RP_INK, textAnchor="middle"))
        d.add(String(cx, y - 10, lab, fontName="Helvetica", fontSize=7,
                     fillColor=RP_MUTED, textAnchor="middle"))


def _chart(r: _Report, summary: pd.DataFrame, joined):
    """Three small panels, each on its own scale.

    A dual axis mixing m/min with PPDA reads badly in print and invites the wrong
    comparison, so the metrics are separated and each gets its own baseline.
    """
    if joined is None or "PPDA" not in joined or joined["PPDA"].isna().all():
        src, has_press = summary, False
    else:
        src, has_press = joined, True

    periods = [p.replace("First half", "H1").replace("Second half", "H2").replace("Extra time", "ET")
               for p in src["Period"]]
    if len(periods) < 2:
        return

    panels = [("Running rate  m/min", [float(v) for v in src["m/min"]], RP_GRASS, "{:.0f}")]
    if "HSR/min" in src:
        panels.append(("High-speed running  m/min", [float(v) for v in src["HSR/min"]], RP_AMBER, "{:.2f}"))
    if "Mechanical load/min" in src:
        panels.append(("Accel + decel  per min", [float(v) for v in src["Mechanical load/min"]], RP_INK, "{:.2f}"))
    if has_press:
        panels.append(("Press intensity  100/PPDA", [100 / float(v) for v in src["PPDA"]], RP_CORAL, "{:.1f}"))

    width = PAGE_W - 2 * MARGIN
    height = 96
    r.ensure(height + 30)
    d = Drawing(width, height)

    gap = 14
    pw = (width - gap * (len(panels) - 1)) / len(panels)
    for i, (title, values, colour, fmt) in enumerate(panels):
        _panel(d, i * (pw + gap), 20, pw, height - 46, title, periods, values, colour, fmt)

    d.drawOn(r.c, MARGIN, r.y - height)
    r.y -= height + 8

    r.caption("Each panel has its own scale. Press intensity is inverted PPDA, so higher means a more aggressive press.")
    r.space(8)


def _player_table(r: _Report, squad: pd.DataFrame, metric="HSR/min", top=8):
    """Per-player rates by period — the part a coach looks for first."""
    if squad is None or squad.empty or metric not in squad:
        return

    pivot = squad.pivot_table(index="Player", columns="Period", values=metric, aggfunc="mean")
    cols = [c for c in ("First half", "Second half", "Extra time") if c in pivot.columns]
    if not cols:
        return
    pivot = pivot[cols]
    if len(cols) >= 2:
        both = pivot[cols[0]].notna() & pivot[cols[1]].notna() & (pivot[cols[0]] != 0)
        pivot["Change"] = float("nan")
        pivot.loc[both, "Change"] = (
            (pivot.loc[both, cols[1]] - pivot.loc[both, cols[0]]) / pivot.loc[both, cols[0]] * 100
        )
        # Players who only featured in one period can't have a change; park them last.
        pivot = pivot.sort_values("Change", ascending=False, na_position="last")
    pivot = pivot.head(top).reset_index()

    r.heading(f"{metric} by player")
    r.caption(f"Ranked by first-to-second-half change. Top {min(top, len(pivot))} of the eligible squad.")

    c = r.c
    usable = PAGE_W - 2 * MARGIN
    name_w = 150
    other_w = (usable - name_w) / max(len(pivot.columns) - 1, 1)

    c.setFillColor(RP_MUTED)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(MARGIN, r.y, "PLAYER")
    for i, col in enumerate(pivot.columns[1:]):
        label = {"First half": "H1", "Second half": "H2", "Extra time": "ET"}.get(col, col)
        c.drawRightString(MARGIN + name_w + (i + 1) * other_w - 6, r.y, label.upper())
    r.y -= 6
    r.rule()
    r.space(9)

    for _, row in pivot.iterrows():
        c.setFillColor(RP_INK)
        c.setFont("Helvetica", 8.5)
        c.drawString(MARGIN, r.y, str(row["Player"])[:32])
        for i, col in enumerate(pivot.columns[1:]):
            v = row[col]
            x = MARGIN + name_w + (i + 1) * other_w - 6
            if col == "Change":
                if pd.isna(v):
                    c.setFillColor(RP_MUTED)
                    c.setFont("Helvetica", 8.5)
                    c.drawRightString(x, r.y, "one period only")
                else:
                    c.setFillColor(RP_GRASS if v >= 0 else RP_CORAL)
                    c.setFont("Helvetica-Bold", 8.5)
                    c.drawRightString(x, r.y, f"{v:+.0f}%")
            else:
                c.setFillColor(RP_INK)
                c.setFont("Helvetica", 8.5)
                c.drawRightString(x, r.y, _fmt(v, 2))
        r.y -= 12.5

    r.space(2)
    r.rule()
    r.space(12)


def _readings(r: _Report, readings):
    if not readings:
        return
    r.heading("Readings")
    for text in readings:
        top = r.y
        r.paragraph(text, READING_STYLE, indent=8)
        r.c.setStrokeColor(RP_GRASS)
        r.c.setLineWidth(2)
        r.c.line(MARGIN + 1, top - 1, MARGIN + 1, r.y + 2)
        r.space(3)
    r.space(6)


def _flags(r: _Report, flags):
    if not flags:
        return
    r.heading("Data quality")
    for kind, msg in flags:
        colour = {"bad": RP_CORAL, "warn": RP_AMBER, "ok": RP_GRASS}.get(kind, RP_MUTED)
        top = r.y
        r.paragraph(msg, BODY_STYLE, indent=8)
        r.c.setStrokeColor(colour)
        r.c.setLineWidth(2)
        r.c.line(MARGIN + 1, top - 1, MARGIN + 1, r.y + 2)
        r.space(3)


def _footer(c: rl_canvas.Canvas, meta: dict, page: int = 1):
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(MARGIN, MARGIN + 16, PAGE_W - MARGIN, MARGIN + 16)
    c.setFillColor(RP_MUTED)
    c.setFont("Helvetica", 7)
    note = "Possession splits are apportioned, not measured."
    left = f"Generated {datetime.now():%d %b %Y, %H:%M}   |   page {page}"
    if not meta.get("sample"):
        left += f"   |   {note}"
    c.drawString(MARGIN, MARGIN + 7, left)
    if meta.get("sample"):
        c.setFillColor(RP_CORAL)
        c.setFont("Helvetica-Bold", 7)
        c.drawRightString(PAGE_W - MARGIN, MARGIN + 7, "SAMPLE DATA - NOT A REAL MATCH")


def build_match_report(summary, joined=None, readings=None, flags=None, meta=None,
                       squad=None) -> bytes:
    """Render the one-page report and return it as PDF bytes."""
    meta = meta or {}
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(meta.get("title", "Match report"))

    r = _Report(c, meta)
    _header(r, meta)
    _metric_strip(r, summary, joined)

    r.heading("Physical output by period")
    r.caption("Exposure-weighted rates. Totals are not comparable when players cover different minutes.")
    _period_table(r, summary, ["Period", "Players", "Total minutes", "Distance (m)",
                               "m/min", "HSR/min", "Mechanical load/min"])
    _chart(r, summary, joined)

    if joined is not None and "Mechanical cost of press" in joined:
        r.heading("Physical output in tactical context")
        _period_table(r, joined, ["Period", "In-possession m/min", "Out-of-possession m/min",
                                  "HSR per unit of press", "Mechanical cost of press",
                                  "xG per 1000 m"])

    _player_table(r, squad)
    _readings(r, readings)
    _flags(r, flags)

    _footer(c, meta, r.page)
    c.showPage()
    c.save()
    return buf.getvalue()

# ==========================================================================
# 4. PITCH VIEWS
# ==========================================================================

PITCH_X, PITCH_Y = 105.0, 68.0

PC_INK = "#12181C"
PC_LINE = "#3A4750"
PC_CHALK = "#C9D6DC"
PC_MUTED = "#8FA3AD"
PC_GRASS = "#4FB477"
PC_AMBER = "#E8A33D"
PC_CORAL = "#E06C5E"


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def draw_pitch(height=560, title=""):
    """An empty pitch, 105 x 68, origin bottom left, attacking left to right."""
    fig = go.Figure()
    shapes = []

    def rect(x0, y0, x1, y1):
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=PC_LINE, width=1.4), layer="below"))

    rect(0, 0, PITCH_X, PITCH_Y)                       # touchlines
    rect(0, 13.84, 16.5, 54.16)                        # left penalty area
    rect(PITCH_X - 16.5, 13.84, PITCH_X, 54.16)        # right penalty area
    rect(0, 24.84, 5.5, 43.16)                         # left six yard
    rect(PITCH_X - 5.5, 24.84, PITCH_X, 43.16)         # right six yard

    shapes.append(dict(type="line", x0=PITCH_X / 2, y0=0, x1=PITCH_X / 2, y1=PITCH_Y,
                       line=dict(color=PC_LINE, width=1.4), layer="below"))
    shapes.append(dict(type="circle", x0=PITCH_X / 2 - 9.15, y0=PITCH_Y / 2 - 9.15,
                       x1=PITCH_X / 2 + 9.15, y1=PITCH_Y / 2 + 9.15,
                       line=dict(color=PC_LINE, width=1.4), layer="below"))

    fig.update_layout(
        shapes=shapes,
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PC_CHALK, size=12),
        xaxis=dict(range=[-3, PITCH_X + 3], visible=False,
                   constrain="domain"),
        yaxis=dict(range=[-3, PITCH_Y + 3], visible=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.04),
    )
    # Attacking direction, so nobody has to ask
    fig.add_annotation(x=PITCH_X / 2, y=-2.2, ax=PITCH_X / 2 - 12, ay=-2.2,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                       arrowcolor=PC_MUTED, text="")
    fig.add_annotation(x=PITCH_X / 2 + 14, y=-2.2, text="attacking", showarrow=False,
                       font=dict(color=PC_MUTED, size=10))
    return fig


# ---------------------------------------------------------------------------
# Normalising direction of play
# ---------------------------------------------------------------------------

def detect_keeper(tracking: pd.DataFrame) -> str:
    """Best guess at the goalkeeper: present all match, and closest to a goal.

    Direction of play is derived from the keeper, so naming the wrong player
    silently corrupts every half-to-half comparison. Auto-detecting a sensible
    default matters more than it looks.
    """
    players = sorted({c[:-2] for c in tracking.columns
                      if c.endswith("_x") and not c.lower().startswith("ball")})
    periods = sorted(tracking["Period"].dropna().unique())

    best, best_score = None, -1.0
    for player in players:
        scores = []
        for period in periods:
            chunk = tracking[tracking["Period"] == period]
            x = chunk[f"{player}_x"].dropna()
            if len(x) < len(chunk) * 0.5:      # not on for most of this half
                scores = []
                break
            scores.append(abs(x.mean() - 0.5))  # distance from halfway
        if scores and np.mean(scores) > best_score:
            best, best_score = player, float(np.mean(scores))
    return best or (players[0] if players else None)


def attacking_direction(tracking: pd.DataFrame, keeper: str,
                        strict: bool = True) -> dict:
    """Work out which way the team attacked in each half, from the keeper.

    The keeper stands nearest his own goal, so his mean x tells you the end the
    team was defending — and therefore the end they were attacking.
    """
    out, missing = {}, []
    for period, chunk in tracking.groupby("Period"):
        x = chunk[f"{keeper}_x"].dropna()
        # Require real coverage. A player who came on at half time gives a reading
        # for one period only, and the other silently falls back to "no flip",
        # which leaves the two halves plotted at opposite ends of the pitch.
        if len(x) < len(chunk) * 0.5:
            missing.append(int(period))
            continue
        # Keeper in the left half means the team attacks right (+1).
        out[int(period)] = 1 if x.mean() < 0.5 else -1

    if strict and missing:
        raise ValueError(
            f"'{keeper}' isn't on the pitch for period(s) {missing}, so the direction "
            "of play can't be established there. Pick a player who played the whole "
            "match — normally the goalkeeper."
        )
    return out


def normalise(x, y, direction):
    """Flip coordinates so the team always attacks left to right."""
    if direction not in (1, -1):
        raise ValueError(
            f"Unknown direction of play ({direction!r}). Refusing to plot rather than "
            "silently leaving a half unflipped."
        )
    if direction == 1:
        return x * PITCH_X, y * PITCH_Y
    return (1 - x) * PITCH_X, (1 - y) * PITCH_Y


# ---------------------------------------------------------------------------
# Average positions, per half
# ---------------------------------------------------------------------------

def average_positions(tracking: pd.DataFrame, keeper: str,
                      min_frames: int = 1500,
                      include_keeper: bool = False) -> pd.DataFrame:
    """Mean position per player per half, direction-normalised.

    The keeper is excluded by default: his position barely moves and including
    him drags the outfield picture toward his goal.
    """
    players = sorted({c[:-2] for c in tracking.columns
                      if c.endswith("_x") and not c.lower().startswith("ball")})
    if not include_keeper:
        players = [p for p in players if p != keeper]
    dirs = attacking_direction(tracking, keeper)

    rows = []
    for period, chunk in tracking.groupby("Period"):
        label = {1: "First half", 2: "Second half"}.get(int(period))
        if label is None or int(period) not in dirs:
            continue
        d = dirs[int(period)]
        if label is None:
            continue
        for player in players:
            x = chunk[f"{player}_x"]
            y = chunk[f"{player}_y"]
            mask = x.notna()
            if mask.sum() < min_frames:
                continue
            nx, ny = normalise(x[mask], y[mask], d)
            rows.append({
                "Player": player,
                "Period": label,
                "x": float(np.mean(nx)),
                "y": float(np.mean(ny)),
                "Frames": int(mask.sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The main story figure
# ---------------------------------------------------------------------------

def shift_map(positions: pd.DataFrame, physical: pd.DataFrame = None,
              metric: str = "HSR/min", title: str = "") -> go.Figure:
    """Where each player stood in each half, with an arrow between.

    Arrow direction is the territorial change. Colour is the change in physical
    output. Together they separate the two stories a squad average cannot:
    a team pushed backwards while working harder, versus a team dropping off.
    """
    fig = draw_pitch(title=title)

    h1 = positions[positions["Period"] == "First half"].set_index("Player")
    h2 = positions[positions["Period"] == "Second half"].set_index("Player")
    shared = [p for p in h1.index if p in h2.index]

    change = {}
    if physical is not None and metric in physical:
        piv = physical.pivot_table(index="Player", columns="Period", values=metric)
        if {"First half", "Second half"}.issubset(piv.columns):
            for player in shared:
                if player in piv.index:
                    a, b = piv.loc[player, "First half"], piv.loc[player, "Second half"]
                    if pd.notna(a) and pd.notna(b) and a:
                        change[player] = (b - a) / a * 100

    # First-half positions, hollow
    fig.add_trace(go.Scatter(
        x=h1.loc[shared, "x"], y=h1.loc[shared, "y"], mode="markers",
        marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=PC_MUTED, width=1.6)),
        name="First half", hovertext=shared, hoverinfo="text",
    ))

    # Arrows: first-half position to second-half position
    for player in shared:
        delta = change.get(player)
        colour = PC_MUTED if delta is None else (PC_GRASS if delta >= 0 else PC_CORAL)
        fig.add_annotation(
            x=h2.loc[player, "x"], y=h2.loc[player, "y"],
            ax=h1.loc[player, "x"], ay=h1.loc[player, "y"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2,
            arrowcolor=colour, opacity=0.85, text="",
        )

    # Second-half positions, filled and labelled
    labels = [p.replace("Player", "") for p in shared]
    hover = [f"{p}<br>{metric} {change[p]:+.0f}%" if p in change else p for p in shared]
    fig.add_trace(go.Scatter(
        x=h2.loc[shared, "x"], y=h2.loc[shared, "y"], mode="markers+text",
        marker=dict(size=26,
                    color=[PC_GRASS if change.get(p, 0) >= 0 else PC_CORAL for p in shared],
                    opacity=0.9, line=dict(color=PC_INK, width=1.5)),
        text=labels, textposition="middle center",
        textfont=dict(color=PC_INK, size=10, family="Arial Black"),
        name="Second half", hovertext=hover, hoverinfo="text",
    ))

    # Defensive line height in each half. Labels are staggered vertically because
    # the two lines sit close together whenever the shift is small — which is
    # exactly when you most need to read both numbers.
    for i, (frame, colour, label) in enumerate(
            ((h1.loc[shared], PC_MUTED, "H1"), (h2.loc[shared], PC_CHALK, "H2"))):
        line_x = frame.nsmallest(4, "x")["x"].mean()   # back four
        fig.add_shape(type="line", x0=line_x, y0=0, x1=line_x, y1=PITCH_Y,
                      line=dict(color=colour, width=1.4, dash="dot"))
        fig.add_annotation(x=line_x, y=PITCH_Y + 1.5 + i * 4.5,
                           text=f"{label} line {line_x:.0f}m",
                           showarrow=False, font=dict(color=colour, size=10))

    return fig


# ---------------------------------------------------------------------------
# Where the ball was won
# ---------------------------------------------------------------------------

def defensive_actions(events: pd.DataFrame, team: str, keeper_dirs: dict,
                      types=("RECOVERY", "CHALLENGE", "INTERCEPTION"),
                      title: str = "") -> go.Figure:
    """Ball recoveries by half, with the mean height of the press marked.

    Press height taken off the pitch is not an opinion. Either the recoveries
    moved up the pitch or they didn't.
    """
    fig = draw_pitch(title=title)

    ev = events[(events["Team"] == team) & (events["Type"].isin(types))].copy()
    ev = ev.dropna(subset=["Start X", "Start Y"])

    palette = {"First half": PC_AMBER, "Second half": PC_GRASS}
    for period, label in ((1, "First half"), (2, "Second half")):
        chunk = ev[ev["Period"] == period]
        if chunk.empty:
            continue
        if period not in keeper_dirs:
            continue
        d = keeper_dirs[period]
        x, y = normalise(chunk["Start X"], chunk["Start Y"], d)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=f"{label}  (n={len(chunk)})",
            marker=dict(size=11, color=palette[label], opacity=0.55,
                        line=dict(color=PC_INK, width=1)),
        ))
        mean_x = float(np.mean(x))
        fig.add_shape(type="line", x0=mean_x, y0=0, x1=mean_x, y1=PITCH_Y,
                      line=dict(color=palette[label], width=2.2, dash="dash"))
        fig.add_annotation(x=mean_x, y=PITCH_Y + 1.6,
                           text=f"{label[:2]} {mean_x:.0f}m", showarrow=False,
                           font=dict(color=palette[label], size=11))
    return fig


# ---------------------------------------------------------------------------
# Where the running happened
# ---------------------------------------------------------------------------

def running_by_zone(tracking: pd.DataFrame, keeper: str, bins_x: int = 6,
                    bins_y: int = 4, hsr_ms: float = 5.5,
                    frame_hz: float = 25.0) -> pd.DataFrame:
    """High-speed distance covered in each pitch zone, per half.

    Answers the question a total cannot: the running fell, but the running
    *where*? Losing high-speed metres in your own third is a different problem
    from losing them in the opposition third.
    """
    players = sorted({c[:-2] for c in tracking.columns
                      if c.endswith("_x") and not c.lower().startswith("ball")
                      and not c.startswith(keeper)})
    dirs = attacking_direction(tracking, keeper)

    rows = []
    for period, chunk in tracking.groupby("Period"):
        label = {1: "First half", 2: "Second half"}.get(int(period))
        if label is None:
            continue
        if int(period) not in dirs:
            continue
        d = dirs[int(period)]
        chunk = chunk.sort_values("Frame")

        for player in players:
            x, y = chunk[f"{player}_x"], chunk[f"{player}_y"]
            if x.notna().sum() < frame_hz * 60:
                continue
            nx, ny = normalise(x, y, d)
            nx = pd.Series(nx).rolling(7, center=True, min_periods=1).mean()
            ny = pd.Series(ny).rolling(7, center=True, min_periods=1).mean()
            step = np.sqrt(nx.diff() ** 2 + ny.diff() ** 2)
            speed = step * frame_hz
            fast = (speed >= hsr_ms) & (speed < 12.0)

            # Drop frames where the player isn't tracked before binning: casting
            # NaN positions to an integer zone index raises, and silently filling
            # them would pile phantom metres into zone (0, 0).
            frame = pd.DataFrame({"x": nx, "y": ny, "m": step.where(fast, 0.0)}).dropna()
            if frame.empty:
                continue
            frame["zx"] = np.clip((frame["x"] / PITCH_X * bins_x).astype(int), 0, bins_x - 1)
            frame["zy"] = np.clip((frame["y"] / PITCH_Y * bins_y).astype(int), 0, bins_y - 1)
            grouped = frame.groupby(["zx", "zy"])["m"].sum().reset_index()
            grouped["Period"] = label
            rows.append(grouped)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows)
    return out.groupby(["Period", "zx", "zy"])["m"].sum().reset_index()


def zone_difference(zones: pd.DataFrame, bins_x: int = 6, bins_y: int = 4,
                    title: str = "") -> go.Figure:
    """Second half minus first half high-speed metres, by zone."""
    fig = draw_pitch(title=title)

    grid = np.zeros((bins_y, bins_x))
    for period, sign in (("First half", -1), ("Second half", 1)):
        chunk = zones[zones["Period"] == period]
        for r in chunk.itertuples():
            grid[int(r.zy), int(r.zx)] += sign * r.m

    limit = np.abs(grid).max() or 1
    cell_w, cell_h = PITCH_X / bins_x, PITCH_Y / bins_y
    for iy in range(bins_y):
        for ix in range(bins_x):
            v = grid[iy, ix]
            t = v / limit
            colour = PC_GRASS if t >= 0 else PC_CORAL
            fig.add_shape(type="rect", layer="below",
                          x0=ix * cell_w, y0=iy * cell_h,
                          x1=(ix + 1) * cell_w, y1=(iy + 1) * cell_h,
                          line=dict(width=0),
                          fillcolor=colour, opacity=min(abs(t) * 0.6, 0.6))
            fig.add_annotation(x=(ix + 0.5) * cell_w, y=(iy + 0.5) * cell_h,
                               text=f"{v:+.0f}", showarrow=False,
                               font=dict(color=PC_CHALK, size=9))
    return fig

# ==========================================================================
# 5. OPEN DATA SOURCES
# ==========================================================================

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# StatsBomb pitch is 120 x 80, always oriented so the acting team attacks +x.
PITCH_LENGTH = 120.0

# PPDA convention: count only the pressing team's attacking 60% of the pitch,
# i.e. exclude their own defensive 40%.
PPDA_LINE = PITCH_LENGTH * 0.40          # 48.0, in the pressing team's frame
OPP_PASS_LINE = PITCH_LENGTH * 0.60      # 72.0, in the passing team's frame

# Conventional PPDA counts tackles, interceptions, challenges and fouls. StatsBomb
# also record "Pressure", an event type no other provider collects — including it
# multiplies the denominator by roughly ten and produces a number that cannot be
# compared with any published PPDA. It is available behind a flag, not by default.
DEFENSIVE_ACTIONS = {"Duel", "Interception", "Foul Committed", "Dribbled Past"}
PRESSURE_ACTION = "Pressure"

PERIOD_NAMES = {1: "First half", 2: "Second half", 3: "Extra time", 4: "Extra time"}


def _get(path):
    url = f"{BASE}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as fh:
            return json.load(fh)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"StatsBomb returned {exc.code} for {path}") from exc


def competitions() -> pd.DataFrame:
    """Every competition and season in the open data set."""
    return pd.DataFrame(_get("competitions.json"))[
        ["competition_id", "season_id", "competition_name", "season_name",
         "competition_gender"]
    ]


def sb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fixtures for one competition season, with a ready-made match label."""
    raw = _get(f"matches/{competition_id}/{season_id}.json")
    rows = []
    for m in raw:
        rows.append({
            "match_id": m["match_id"],
            "date": m["match_date"],
            "home": m["home_team"]["home_team_name"],
            "away": m["away_team"]["away_team_name"],
            "score": f"{m.get('home_score', '')}-{m.get('away_score', '')}",
            "competition": m["competition"]["competition_name"],
            "season": m["season"]["season_name"],
        })
    out = pd.DataFrame(rows).sort_values("date")
    out["label"] = out["home"] + " v " + out["away"] + "  (" + out["date"] + ")"
    return out


def events(match_id: int) -> pd.DataFrame:
    """Raw event stream for one match."""
    return pd.json_normalize(_get(f"events/{match_id}.json"), sep="_")


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def period_analytics(match_id: int, team: str = None,
                     include_pressure: bool = False) -> pd.DataFrame:
    """Per-half xG, PPDA, possession and field tilt, in the app's column format.

    `team` is the side the report is about. Defaults to the home team.

    `include_pressure` adds StatsBomb's Pressure events to the PPDA denominator.
    It measures something real — pressing volume — but the result is roughly five
    times lower than conventional PPDA and comparable only with itself.
    """
    action_types = set(DEFENSIVE_ACTIONS) | ({PRESSURE_ACTION} if include_pressure else set())
    ev = events(match_id)

    if team is None:
        # The first event's possession team is the side that kicked off; use the
        # home team instead, which is the first team named in the lineups.
        team = ev.loc[ev["type_name"] == "Starting XI", "team_name"].iloc[0]

    teams = list(ev["team_name"].dropna().unique())
    if team not in teams:
        raise ValueError(f"'{team}' not in this match. Teams are: {teams}")
    opponent = [t for t in teams if t != team][0]

    rows = []
    for period, chunk in ev.groupby("period"):
        label = PERIOD_NAMES.get(int(period))
        if label is None:
            continue

        shots = chunk[chunk["type_name"] == "Shot"]
        xg_col = "shot_statsbomb_xg"
        xg_for = shots.loc[shots["team_name"] == team, xg_col].sum() if xg_col in shots else 0.0
        xg_against = shots.loc[shots["team_name"] == opponent, xg_col].sum() if xg_col in shots else 0.0

        passes = chunk[chunk["type_name"] == "Pass"]
        our_passes = int((passes["team_name"] == team).sum())
        their_passes = int((passes["team_name"] == opponent).sum())

        # Possession share by pass volume. StatsBomb also ship a possession
        # counter, but pass share is stable and needs no extra assumptions.
        possession = _safe_ratio(our_passes, our_passes + their_passes) * 100

        # Field tilt: share of final-third passes. A better read of territorial
        # control than raw possession, because it ignores sterile circulation.
        final_third = passes[passes["location"].apply(
            lambda loc: isinstance(loc, list) and len(loc) >= 1 and loc[0] >= 80)]
        ours_ft = int((final_third["team_name"] == team).sum())
        theirs_ft = int((final_third["team_name"] == opponent).sum())
        tilt = _safe_ratio(ours_ft, ours_ft + theirs_ft) * 100

        def ppda(pressing, passing):
            """Opposition passes allowed per defensive action, in the press zone."""
            opp_passes = passes[
                (passes["team_name"] == passing)
                & passes["location"].apply(
                    lambda loc: isinstance(loc, list) and loc[0] <= OPP_PASS_LINE)
            ]
            actions = chunk[
                (chunk["team_name"] == pressing)
                & chunk["type_name"].isin(action_types)
                & chunk["location"].apply(
                    lambda loc: isinstance(loc, list) and loc[0] >= PPDA_LINE)
            ]
            return _safe_ratio(len(opp_passes), len(actions))

        rows.append({
            "Period": label,
            "xG for": round(float(xg_for), 3),
            "xG against": round(float(xg_against), 3),
            "PPDA": round(ppda(team, opponent), 2),
            "Opp PPDA": round(ppda(opponent, team), 2),
            "Possession %": round(possession, 1),
            "Field tilt %": round(tilt, 1),
        })

    order = {"First half": 0, "Second half": 1, "Extra time": 2}
    out = pd.DataFrame(rows).sort_values("Period", key=lambda s: s.map(order))
    out.attrs["team"] = team
    out.attrs["opponent"] = opponent
    return out


def analytics_csv(jobs) -> pd.DataFrame:
    """Build a multi-match analytics file.

    `jobs` is a list of (match_label, match_id, team). match_label must match the
    label the GPS side produces, e.g. '04/09/2026 v Denmark', so the app can join
    the two. Returns a frame ready to save as CSV and upload.
    """
    frames = []
    for label, match_id, team in jobs:
        rows = period_analytics(match_id, team)
        rows.insert(0, "Match", label)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrica Sports open data — tracking AND events for the same matches
# ---------------------------------------------------------------------------
#
# Metrica released three anonymised matches with 25 Hz tracking for every player
# plus the matching event stream. Tracking is the only free source from which
# genuine physical metrics can be derived — distance, speed, high-speed running,
# accelerations — because you compute them yourself from positions rather than
# relying on a provider's published aggregate.
#
# It is the one open source that fills BOTH halves of this app's join.
#
# Caveat that matters: these are optical positions, not GPS pods. Distances will
# not equal StatSports values for the same match, and the two should never be
# plotted on one axis. Use it to build and demo the tool, not to benchmark a club.

METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

PITCH_X, PITCH_Y = 105.0, 68.0   # metres, Metrica convention
FRAME_HZ = 25.0

# Thresholds in m/s. 5.5 m/s is ~19.8 km/h, the common high-speed running line;
# 7.0 m/s is ~25.2 km/h for sprinting. Clubs set these differently — change them
# to match whatever the club's StatSports profile uses before comparing anything.
HSR_MS = 5.5
SPRINT_MS = 7.0
ACCEL_MS2 = 3.0


def metrica_events(game: int = 1) -> pd.DataFrame:
    """Event stream for one Metrica sample game (1, 2 or 3)."""
    return pd.read_csv(f"{METRICA}/Sample_Game_{game}/Sample_Game_{game}_RawEventsData.csv")


class TrackingFormatError(ValueError):
    """Raised when a file doesn't look like positional tracking data."""


def tidy_tracking(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a wide tracking export into Period / Frame / Time / player_x,_y.

    Used by both the Metrica downloader and the app's upload path, so an uploaded
    file goes through exactly the same normalisation as a fetched one.
    """
    cols = list(raw.columns)
    if len(cols) < 5:
        raise TrackingFormatError(
            "This file has too few columns to be tracking data. Tracking has one x and "
            "one y column per player. If you meant to load a StatSports drill export, "
            "that belongs in the sidebar uploader — it has no coordinates, so it cannot "
            "drive the pitch views."
        )

    # Validate BEFORE renaming. The rename below manufactures the _x/_y suffixes,
    # so checking for them afterwards would pass any wide CSV, including a GPS
    # drill export whose columns would be silently relabelled as coordinates.
    already_tidy = any(str(c).endswith("_x") for c in cols)
    if not already_tidy:
        # In a Metrica-layout file every y column is unheaded, so roughly half the
        # columns past the first three read as "Unnamed".
        tail = cols[3:]
        unnamed = sum(1 for c in tail if str(c).startswith("Unnamed"))
        if not tail or unnamed / len(tail) < 0.35:
            raise TrackingFormatError(
                "This doesn't look like positional tracking data. A tracking export "
                "pairs each player with an unheaded second column for the y coordinate. "
                "A StatSports drill export has one row per player per drill and no "
                "coordinates at all — it can't drive the pitch views. Load it in the "
                "sidebar instead."
            )

        numeric = raw[tail].apply(pd.to_numeric, errors="coerce")
        if numeric.notna().mean().mean() < 0.5:
            raise TrackingFormatError(
                "The coordinate columns are mostly non-numeric, so this can't be "
                "tracking data. Check the file has the two-row Metrica header above "
                "the data."
            )

    # Columns arrive as Player11, Unnamed:4, Player1, Unnamed:6 ... — x then y.
    renamed = {cols[0]: "Period", cols[1]: "Frame", cols[2]: "Time"}
    for i in range(3, len(cols) - 1, 2):
        name = str(cols[i])
        if name.startswith("Unnamed"):
            continue
        renamed[cols[i]] = f"{name}_x"
        renamed[cols[i + 1]] = f"{name}_y"
    out = raw.rename(columns=renamed)

    coord_cols = [c for c in out.columns if c.endswith(("_x", "_y"))]
    if not coord_cols:
        raise TrackingFormatError(
            "No player coordinate columns found. Expected a header row naming each "
            "player, with an x column and a y column for each. Check the file has the "
            "two-row Metrica header, and that you haven't uploaded a GPS drill export."
        )
    if "Period" not in out.columns:
        raise TrackingFormatError(
            "No Period column found. The first three columns should be Period, Frame "
            "and Time. Check whether this export uses a different header layout."
        )

    return out[["Period", "Frame", "Time"] + coord_cols]


def metrica_tracking(game: int = 1, side: str = "Home") -> pd.DataFrame:
    """25 Hz tracking for one side, tidied into Period / Frame / Time / player x,y."""
    url = f"{METRICA}/Sample_Game_{game}/Sample_Game_{game}_RawTrackingData_{side}_Team.csv"
    return tidy_tracking(pd.read_csv(url, skiprows=2))


def metrica_physical(game: int = 1, side: str = "Home",
                     smooth_frames: int = 7) -> pd.DataFrame:
    """Per-player, per-half physical metrics derived from tracking positions.

    Returns the same shape of metrics the GPS pipeline produces, so the output can
    be compared like for like: minutes, distance, m/min, HSR, sprints, accels.
    """
    track = metrica_tracking(game, side)
    return metrica_physical_from_frame(track, smooth_frames=smooth_frames)


def metrica_physical_from_frame(track: pd.DataFrame, keeper: str = None,
                                smooth_frames: int = 7) -> pd.DataFrame:
    """Same as metrica_physical() but from a tracking frame already in memory.

    Lets the app compute once from an uploaded or cached file instead of
    re-downloading. `keeper` is accepted so callers can pass it positionally
    alongside the tracking frame; it does not change the calculation.
    """
    # The tracking file carries the ball as another tracked object. Left in, it
    # contributes ~420 m/min and wrecks every squad-level rate.
    players = sorted({c[:-2] for c in track.columns
                      if c.endswith("_x") and not c.lower().startswith("ball")})

    rows = []
    for period, chunk in track.groupby("Period"):
        chunk = chunk.sort_values("Frame")
        label = PERIOD_NAMES.get(int(period))
        if label is None:
            continue
        minutes = len(chunk) / FRAME_HZ / 60

        for player in players:
            x = chunk[f"{player}_x"] * PITCH_X
            y = chunk[f"{player}_y"] * PITCH_Y
            if x.notna().sum() < FRAME_HZ * 60:      # under a minute on pitch
                continue

            # Smooth before differencing: raw optical positions are noisy, and
            # unsmoothed velocity badly overstates distance and acceleration.
            xs = x.rolling(smooth_frames, center=True, min_periods=1).mean()
            ys = y.rolling(smooth_frames, center=True, min_periods=1).mean()

            step = np.sqrt(xs.diff() ** 2 + ys.diff() ** 2)
            speed = step * FRAME_HZ                                   # m/s
            speed = speed.where(speed < 12.0)                         # drop tracking jumps
            accel = speed.diff() * FRAME_HZ                           # m/s^2

            on_pitch_min = x.notna().sum() / FRAME_HZ / 60
            distance = float(step.sum())
            hsr = float(step[speed >= HSR_MS].sum())
            sprint = float(step[speed >= SPRINT_MS].sum())

            # Count efforts, not frames: a run is one effort however long it lasts.
            def efforts(mask):
                return int((mask.astype(int).diff() == 1).sum())

            rows.append({
                "Player": player,
                "Period": label,
                "Minutes": round(on_pitch_min, 2),
                "Distance (m)": round(distance, 1),
                "m/min": round(distance / on_pitch_min, 1) if on_pitch_min else np.nan,
                "HSR (m)": round(hsr, 1),
                "HSR/min": round(hsr / on_pitch_min, 2) if on_pitch_min else np.nan,
                "Sprint distance (m)": round(sprint, 1),
                "Sprints": efforts(speed >= SPRINT_MS),
                "Accels": efforts(accel >= ACCEL_MS2),
                "Decels": efforts(accel <= -ACCEL_MS2),
                "Max speed (m/s)": round(float(speed.max()), 2),
            })

    order = {"First half": 0, "Second half": 1, "Extra time": 2}
    out = pd.DataFrame(rows)
    return out.sort_values(["Player", "Period"], key=lambda s: s.map(order) if s.name == "Period" else s)


def metrica_period_summary(game: int = 1, side: str = "Home",
                           exclude: tuple = ()) -> pd.DataFrame:
    """Squad-level exposure-weighted rates per half, matching the app's summary.

    `exclude` drops named players — worth using for the goalkeeper, whose ~40 m/min
    pulls the squad rate down and isn't comparable to outfield work anyway.
    """
    phys = metrica_physical(game, side)
    if exclude:
        phys = phys[~phys["Player"].isin(exclude)]
    rows = []
    for period, g in phys.groupby("Period"):
        mins = g["Minutes"].sum()
        rows.append({
            "Period": period,
            "Players": g["Player"].nunique(),
            "Total minutes": round(mins, 1),
            "Distance (m)": round(g["Distance (m)"].sum(), 1),
            "m/min": round(g["Distance (m)"].sum() / mins, 1),
            "HSR/min": round(g["HSR (m)"].sum() / mins, 2),
            "Mechanical load/min": round((g["Accels"].sum() + g["Decels"].sum()) / mins, 2),
        })
    order = {"First half": 0, "Second half": 1, "Extra time": 2}
    return pd.DataFrame(rows).sort_values("Period", key=lambda s: s.map(order))


def metrica_as_statsports(game: int = 1, side: str = "Home", opponent: str = "Away",
                          date: str = "01/01/2020", exclude: tuple = ()) -> pd.DataFrame:
    """Write Metrica-derived physical data in the column layout the app ingests.

    Lets you drive the whole tool from open data end to end: this for the GPS
    side, period_analytics() for the analytics side, joined on the same periods.
    Values are optical-derived, so label them as such wherever they are shown.
    """
    phys = metrica_physical(game, side)
    if exclude:
        phys = phys[~phys["Player"].isin(exclude)]

    drill = {"First half": f"2. First Half v {opponent}",
             "Second half": f"3. Second Half v {opponent}",
             "Extra time": f"4. Extra Time v {opponent}"}

    def hms(minutes):
        total = int(round(minutes * 60))
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    out = pd.DataFrame({
        "Player Display Name": phys["Player"],
        "Date": date,
        "Drill Title": phys["Period"].map(drill),
        "Player Primary Position": "Unknown",
        "Total Time": phys["Minutes"].map(hms),
        "Total Distance": phys["Distance (m)"],
        "Distance Per Min": phys["m/min"],
        "High Speed Running (Absolute)": phys["HSR (m)"],
        "HSR Per Minute (Absolute)": phys["HSR/min"],
        "Sprints": phys["Sprints"],
        "Sprint Distance": phys["Sprint distance (m)"],
        "Accelerations (Absolute)": phys["Accels"],
        "Decelerations (Absolute)": phys["Decels"],
        "Max Speed": (phys["Max speed (m/s)"] * 3.6).round(2),   # to km/h
    })
    return out.reset_index(drop=True)

# ==========================================================================
# 6. THE APP
# ==========================================================================

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Match Context", page_icon="◐", layout="wide")

INK = "#12181C"
SLATE = "#1E272E"
LINE = "#2E3A42"
PAPER = "#E8EDEF"
MUTED = "#8FA3AD"
GRASS = "#4FB477"
AMBER = "#E8A33D"
CORAL = "#E06C5E"
CHALK = "#C9D6DC"

st.markdown(
    f"""
    <style>
      .stApp {{ background: {INK}; color: {PAPER}; }}
      section[data-testid="stSidebar"] {{ background: {SLATE}; border-right: 1px solid {LINE}; }}
      h1, h2, h3, h4 {{ color: {PAPER}; letter-spacing: -0.01em; font-weight: 600; }}
      h1 {{ font-size: 1.85rem; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {LINE}; }}
      .stTabs [data-baseweb="tab"] {{ background: transparent; color: {MUTED}; padding: 0 0 0.6rem 0; }}
      .stTabs [aria-selected="true"] {{ color: {PAPER}; border-bottom: 2px solid {GRASS}; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.6rem; color: {PAPER}; }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      .lede {{ color: {MUTED}; font-size: 0.95rem; margin: -0.4rem 0 1.6rem 0; }}
      .flag {{ border-left: 3px solid {AMBER}; padding: 0.55rem 0.9rem; margin: 0.4rem 0;
              background: rgba(232,163,61,0.07); font-size: 0.9rem; }}
      .flag-bad {{ border-left-color: {CORAL}; background: rgba(224,108,94,0.08); }}
      .flag-ok {{ border-left-color: {GRASS}; background: rgba(79,180,119,0.07); }}
      .note {{ color: {MUTED}; font-size: 0.82rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def shade(series_min, series_max):
    """Return a Styler function colouring cells low-to-high.

    Written by hand rather than using Styler.background_gradient, which pulls in
    matplotlib purely to interpolate a colour map — a heavy dependency for a
    gradient, and one more thing to fail on deploy.
    """
    span = (series_max - series_min) or 1.0
    lo = (0xE0, 0x6C, 0x5E)   # coral
    mid = (0xE8, 0xA3, 0x3D)  # amber
    hi = (0x4F, 0xB4, 0x77)   # grass

    def colour(value):
        if value is None or pd.isna(value):
            return "color: #8FA3AD"
        t = min(max((float(value) - series_min) / span, 0.0), 1.0)
        a, b, local = (lo, mid, t / 0.5) if t < 0.5 else (mid, hi, (t - 0.5) / 0.5)
        rgb = tuple(int(a[i] + (b[i] - a[i]) * local) for i in range(3))
        # Keep the fill translucent so the dark theme still reads underneath.
        text = "#0E1417" if t > 0.55 else "#E8EDEF"
        return f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},0.55); color: {text}"

    return colour


PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=CHALK, size=12),
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ---------------------------------------------------------------------------
# Pipeline (pure logic lives in pipeline.py so it can be tested and reused)
# ---------------------------------------------------------------------------



@st.cache_data(show_spinner=False)
def cached_load_gps(file):
    return load_gps(file)


@st.cache_data(show_spinner=False)
def cached_tracking(game, side):
    return metrica_tracking(game, side)


@st.cache_data(show_spinner=False)
def cached_events(game):
    return metrica_events(game)


@st.cache_data(show_spinner=False)
def cached_physical(tracking, keeper):
    """Per-player physical from tracking. Cached: it is the slow step."""
    return metrica_physical_from_frame(tracking, keeper)


@st.cache_data(show_spinner=False)
def cached_zones(tracking, keeper):
    return running_by_zone(tracking, keeper)


@st.cache_data(show_spinner=False)
def load_library(payloads):
    """payloads: tuple of (name, bytes). Hashable, so Streamlit can cache it."""
    return load_many([(name, io.BytesIO(data)) for name, data in payloads])


def period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Squad rates per period, exposure-weighted."""
    rows = []
    for period, g in df.groupby("Period"):
        total_min = g["Minutes"].sum()
        row = {
            "Period": period,
            "Players": g["Player"].nunique(),
            "Total minutes": total_min,
            "Distance (m)": g["Distance (m)"].sum(),
            "m/min": g["Distance (m)"].sum() / total_min if total_min else np.nan,
        }
        if "HSR (m)" in g:
            row["HSR (m)"] = g["HSR (m)"].sum()
            row["HSR/min"] = g["HSR (m)"].sum() / total_min if total_min else np.nan
        if "Accels" in g and "Decels" in g:
            row["Mechanical load/min"] = (
                (g["Accels"].sum() + g["Decels"].sum()) / total_min if total_min else np.nan
            )
        if "Sprint distance (m)" in g:
            row["Sprint distance (m)"] = g["Sprint distance (m)"].sum()
        if "Max speed" in g:
            row["Peak speed"] = g["Max speed"].max()
        rows.append(row)
    order = {p: i for i, p in enumerate(MATCH_PERIODS)}
    return pd.DataFrame(rows).sort_values("Period", key=lambda s: s.map(order))


# ---------------------------------------------------------------------------
# Sidebar — data in
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Data")
gps_files = st.sidebar.file_uploader(
    "StatSports exports (CSV)", type=["csv"], accept_multiple_files=True,
    help="Upload one match or several. Matches are keyed on the date and the opponent "
         "read from the drill titles.",
)

if not gps_files:
    st.title("Match Context")
    st.markdown(
        '<p class="lede">Physical output read against what the team was actually doing on the pitch.</p>',
        unsafe_allow_html=True,
    )
    st.info("Upload one or more StatSports drill exports to begin. Any export with player name, "
            "drill title and total distance will parse.")
    st.markdown(
        """
**What this does**

Your GPS export is split by drill — warm up, first half, second half, conditioning. Match analytics
are reported the same way: per half, per phase. This tool joins the two on that shared key and shows
physical output alongside the tactical picture that produced it.

Upload several matches and it will also track how the squad and each player move across them.

**What it does not do**

It won't give you minute-by-minute synchronisation. That needs raw 10 Hz files and timestamped event
data, which is a different build.
        """
    )
    st.stop()

# Read once into bytes so the loader can be cached across reruns.
payloads = tuple((f.name, f.getvalue()) for f in gps_files)
library = load_library(payloads)

if library.empty:
    st.error("Nothing parsed from those files. Check they are StatSports drill exports.")
    for name, msg in library.attrs.get("errors", []):
        st.caption(f"{name}: {msg}")
    st.stop()

for name, msg in library.attrs.get("errors", []):
    st.sidebar.warning(f"Skipped {name} — {msg}")

matches = ordered_matches(library)

st.sidebar.markdown("### Match")
if len(matches) > 1:
    selected_match = st.sidebar.selectbox(
        "Report on", matches, index=len(matches) - 1,
        help="The first four tabs cover this match. The last tab compares across all of them.",
    )
else:
    selected_match = matches[0]
    st.sidebar.caption(matches[0])

gps = library[library["Match"] == selected_match].copy()
periods_present = [p for p in MATCH_PERIODS if p in set(gps["Period"])]

st.sidebar.markdown("### Filters")
positions = sorted(library["Position"].unique())
sel_positions = st.sidebar.multiselect("Position", positions, default=positions)

min_exposure = st.sidebar.slider("Minimum match minutes", 0, 60, 15, 5,
                                 help="Players below this are excluded from squad comparisons.")

library = library[library["Position"].isin(sel_positions)]
gps = gps[gps["Position"].isin(sel_positions)]

# ---------------------------------------------------------------------------
# Analytics input — period level
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Match analytics")
st.sidebar.caption(
    "Upload one CSV covering every match (add a Match column), or type figures in "
    "the second tab. Entries are kept per match while the app is open."
)

analytics_file = st.sidebar.file_uploader("Analytics CSV (optional)", type=["csv"], key="an")

ANALYTICS_COLS = ["xG for", "xG against", "PPDA", "Opp PPDA", "Possession %", "Field tilt %"]


def blank_analytics(periods):
    periods = list(periods) or ["First half", "Second half"]
    frame = pd.DataFrame({"Period": periods})
    for col in ANALYTICS_COLS:
        frame[col] = 10.0 if "PPDA" in col else (50.0 if "%" in col else 0.0)
    return frame


# Analytics are held per match. Without this, switching match in the sidebar would
# carry one game's xG and PPDA onto another and quietly produce a wrong report.
if "analytics_by_match" not in st.session_state:
    st.session_state.analytics_by_match = {}

uploaded_analytics = None
if analytics_file is not None:
    try:
        uploaded_analytics = pd.read_csv(analytics_file)
        uploaded_analytics.columns = [c.strip() for c in uploaded_analytics.columns]
    except Exception as exc:
        st.sidebar.error(f"Couldn't read that analytics CSV: {exc}")

if uploaded_analytics is not None:
    if "Match" in uploaded_analytics.columns:
        matched = 0
        for label, rows in uploaded_analytics.groupby("Match"):
            label = str(label).strip()
            if label in matches:
                st.session_state.analytics_by_match[label] = (
                    rows.drop(columns=["Match"]).reset_index(drop=True)
                )
                matched += 1
        unknown = sorted(
            set(uploaded_analytics["Match"].astype(str).str.strip()) - set(matches)
        )
        st.sidebar.success(f"Analytics loaded for {matched} of {len(matches)} match(es).")
        if unknown:
            st.sidebar.caption(
                "No GPS file carries these labels: " + ", ".join(unknown[:3])
                + ("..." if len(unknown) > 3 else "")
            )
    else:
        # No Match column, so it can only describe the match currently selected.
        st.session_state.analytics_by_match[selected_match] = uploaded_analytics.copy()
        st.sidebar.info(f"No Match column — applied to {selected_match}.")

analytics = st.session_state.analytics_by_match.get(
    selected_match, blank_analytics(periods_present)
)

st.title("Match Context")
date_label = str(gps["Date"].iloc[0]) if "Date" in gps and len(gps) else ""
lib_note = f" · {len(matches)} matches loaded" if len(matches) > 1 else ""
st.markdown(
    f'<p class="lede">{selected_match}{lib_note}</p>',
    unsafe_allow_html=True,
)

tab_names = ["Period profile", "Physical in context", "Players", "Pitch", "Data check", "Export"]
if len(matches) > 1:
    tab_names.insert(4, "Across matches")
tabs = st.tabs(tab_names)
IDX = {name: i for i, name in enumerate(tab_names)}

# Populated inside the context tab; the Export tab reads them further down.
joined = None
readings = []


# ---------------------------------------------------------------------------
# Squad-level period aggregation
# ---------------------------------------------------------------------------

match = gps[gps["Period"].isin(MATCH_PERIODS)].copy()
eligible = match.groupby("Player")["Minutes"].sum()
eligible = eligible[eligible >= min_exposure].index
squad = match[match["Player"].isin(eligible)].copy()


summary = period_summary(squad) if len(squad) else pd.DataFrame()

# ---------------------------------------------------------------------------
# TAB 1 — Period profile
# ---------------------------------------------------------------------------

with tabs[IDX["Period profile"]]:
    if summary.empty:
        st.warning("No match periods found. Check the drill titles in your export name the halves.")
    else:
        cols = st.columns(len(summary))
        for col, (_, row) in zip(cols, summary.iterrows()):
            with col:
                st.metric(
                    row["Period"],
                    f"{row['m/min']:.1f} m/min",
                    f"{row['Total minutes']:.0f} player-minutes",
                    delta_color="off",
                )

        st.markdown("#### Rate metrics by period")
        st.caption("All exposure-weighted. Totals are not comparable when players cover different minutes.")

        rate_cols = [c for c in ["m/min", "HSR/min", "Mechanical load/min"] if c in summary]
        fig = go.Figure()
        palette = [GRASS, AMBER, CORAL]
        for i, metric in enumerate(rate_cols):
            fig.add_bar(
                name=metric,
                x=summary["Period"],
                y=summary[metric],
                marker_color=palette[i % len(palette)],
                text=summary[metric].round(2),
                textposition="outside",
            )
        fig.update_layout(barmode="group", height=380, title="", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

        # Half-to-half change
        if {"First half", "Second half"}.issubset(set(summary["Period"])):
            h1 = summary[summary["Period"] == "First half"].iloc[0]
            h2 = summary[summary["Period"] == "Second half"].iloc[0]
            st.markdown("#### First half to second half")
            deltas = st.columns(len(rate_cols))
            for col, metric in zip(deltas, rate_cols):
                change = (h2[metric] - h1[metric]) / h1[metric] * 100 if h1[metric] else 0
                col.metric(metric, f"{h2[metric]:.2f}", f"{change:+.1f}%")
            st.markdown(
                '<p class="note">A fall in rate metrics across halves is normal. '
                "It only becomes a finding when it moves differently from the tactical picture — see the next tab.</p>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# TAB 2 — Physical in context
# ---------------------------------------------------------------------------

with tabs[IDX["Physical in context"]]:
    st.markdown("#### Match analytics")
    st.caption("Edit these to match your Opta / Wyscout / StatsBomb period splits. Values persist while the app runs.")

    edited = st.data_editor(
        analytics,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"analytics_editor::{selected_match}",
    )
    # Persist so the figures are still here after switching match and back.
    st.session_state.analytics_by_match[selected_match] = edited

    if summary.empty:
        st.warning("No GPS match periods to join against.")
    else:
        joined = summary.merge(edited, on="Period", how="left")

        if joined[["PPDA", "Possession %"]].isna().all().all():
            st.info("Enter analytics values above to see the contextual metrics.")
        else:
            # Contextual derivations
            joined["Out-of-possession m/min"] = joined["m/min"] * (1 - joined["Possession %"] / 100)
            joined["In-possession m/min"] = joined["m/min"] * (joined["Possession %"] / 100)
            # PPDA is inverted by definition: a LOWER value means a MORE aggressive press.
            # Convert it to an intensity that rises with pressure before dividing by it,
            # otherwise "cost per unit of press" comes out with the sign reversed.
            joined["Press intensity"] = 100 / joined["PPDA"]
            joined["HSR per unit of press"] = joined["HSR/min"] / joined["Press intensity"]
            joined["xG per 1000 m"] = joined["xG for"] / (joined["Distance (m)"] / 1000)
            joined["Mechanical cost of press"] = joined["Mechanical load/min"] / joined["Press intensity"]

            st.markdown("#### Physical output against tactical picture")

            fig = go.Figure()
            fig.add_bar(
                x=joined["Period"], y=joined["m/min"], name="m/min",
                marker_color=GRASS, yaxis="y", opacity=0.85,
            )
            fig.add_trace(
                go.Scatter(
                    x=joined["Period"], y=joined["PPDA"], name="PPDA",
                    yaxis="y2", mode="lines+markers",
                    line=dict(color=AMBER, width=3), marker=dict(size=11),
                )
            )
            layout = dict(PLOT_LAYOUT)
            layout.pop("yaxis")
            fig.update_layout(
                height=400,
                yaxis=dict(title="m/min", gridcolor=LINE),
                yaxis2=dict(title="PPDA (lower = more aggressive press)",
                            overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
                **layout,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Derived measures")
            show = [
                "Period", "m/min", "In-possession m/min", "Out-of-possession m/min",
                "HSR per unit of press", "Mechanical cost of press", "xG per 1000 m",
            ]
            show = [c for c in show if c in joined]
            st.dataframe(joined[show].round(3), use_container_width=True, hide_index=True)

            st.markdown(
                '<p class="note">In / out of possession splits are apportioned by possession share, not measured '
                "directly. They are a reasonable proxy at period level; treat them as directional. Measured splits "
                "need event-synchronised raw files.</p>",
                unsafe_allow_html=True,
            )

            # Readings
            st.markdown("#### Readings")
            readings = []
            if {"First half", "Second half"}.issubset(set(joined["Period"])):
                a = joined[joined["Period"] == "First half"].iloc[0]
                b = joined[joined["Period"] == "Second half"].iloc[0]

                pct = lambda x, y: (y - x) / x * 100 if x else 0.0
                d_rate = pct(a["m/min"], b["m/min"])
                d_ppda = pct(a["PPDA"], b["PPDA"])
                d_hsr = pct(a.get("HSR/min", 0), b.get("HSR/min", 0))
                d_mech = pct(a.get("Mechanical load/min", 0), b.get("Mechanical load/min", 0))

                # Remember: PPDA down = press intensified.
                pressed_more = d_ppda < -8
                pressed_less = d_ppda > 8

                if pressed_more and d_hsr > 10:
                    readings.append(
                        f"The team went after the game. Press intensified ({d_ppda:.0f}% PPDA) and high-speed "
                        f"running rose {d_hsr:.0f}%, while total running rate moved {d_rate:+.0f}%. They didn't run "
                        "further, they ran harder — a change in the type of work, not the amount."
                    )
                elif pressed_more:
                    readings.append(
                        f"Press intensified ({d_ppda:.0f}% PPDA) but high-speed running moved only {d_hsr:+.0f}%. "
                        "More pressure asked for without more intensity behind it. Worth checking whether the press "
                        "was being triggered but not supported."
                    )
                elif pressed_less and d_rate < -5:
                    readings.append(
                        f"Running rate fell {abs(d_rate):.0f}% and the press loosened ({d_ppda:+.0f}% PPDA). "
                        "Consistent picture — the team dropped off and the physical numbers follow. Whether that "
                        "was chosen or forced is a coaching conversation, not a data one."
                    )
                elif pressed_less and d_rate > 5:
                    readings.append(
                        f"Running rate rose {d_rate:.0f}% while the press loosened. Output went into recovery runs "
                        "and transitions rather than pressure — often a chasing-the-game profile."
                    )
                else:
                    readings.append(
                        f"Running rate moved {d_rate:+.0f}%, high-speed running {d_hsr:+.0f}% and PPDA "
                        f"{d_ppda:+.0f}%. Broadly stable across halves."
                    )

                if "Mechanical cost of press" in joined:
                    d_cost = pct(a["Mechanical cost of press"], b["Mechanical cost of press"])
                    if d_cost > 15:
                        readings.append(
                            f"Accel/decel load per unit of press rose {d_cost:.0f}%. The same pressure is being "
                            "bought with more braking — that is the load that shows up in the following days."
                        )
                    elif d_cost < -15:
                        readings.append(
                            f"Accel/decel load per unit of press fell {abs(d_cost):.0f}%. More press for less "
                            f"mechanical outlay, even though absolute load rose {d_mech:+.0f}%. Efficient, but the "
                            "absolute figure is still what the players have to recover from."
                        )

            for r in readings:
                st.markdown(f'<div class="flag flag-ok">{r}</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="note">These are prompts for the debrief, not conclusions. One match is one match.</p>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# TAB 3 — Players
# ---------------------------------------------------------------------------

with tabs[IDX["Players"]]:
    if match.empty:
        st.warning("No match periods in this export.")
    else:
        metric_options = [c for c in ["m/min", "HSR/min", "Mechanical load/min", "Max speed",
                                      "Metabolic power", "Speed intensity"] if c in match]
        metric = st.selectbox("Metric", metric_options, index=0)

        st.markdown(f"#### {metric} by player and period")
        st.caption(f"Players with at least {min_exposure} match minutes. Rate metrics, so exposure differences are handled.")

        pivot = squad.pivot_table(index="Player", columns="Period", values=metric, aggfunc="mean")
        pivot = pivot.reindex(columns=[p for p in MATCH_PERIODS if p in pivot.columns])
        pivot = pivot.sort_values(pivot.columns[-1], ascending=True)

        fig = go.Figure()
        palette = {"First half": GRASS, "Second half": AMBER, "Extra time": CORAL}
        for period in pivot.columns:
            fig.add_bar(
                y=pivot.index, x=pivot[period], name=period, orientation="h",
                marker_color=palette.get(period, MUTED),
            )
        fig.update_layout(barmode="group", height=max(360, 42 * len(pivot)), **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Individual")
        player = st.selectbox("Player", sorted(match["Player"].unique()))
        pdata = gps[gps["Player"] == player]
        pmatch = pdata[pdata["Period"].isin(MATCH_PERIODS)]

        c = st.columns(4)
        c[0].metric("Match minutes", f"{pmatch['Minutes'].sum():.0f}")
        c[1].metric("Match distance", f"{pmatch['Distance (m)'].sum():,.0f} m")
        if pmatch["Minutes"].sum():
            c[2].metric("m/min", f"{pmatch['Distance (m)'].sum() / pmatch['Minutes'].sum():.1f}")
        if "Max speed" in pmatch:
            c[3].metric("Peak speed", f"{pmatch['Max speed'].max():.2f}")

        display_cols = [c for c in ["Drill", "Minutes", "Distance (m)", "m/min", "HSR (m)", "HSR/min",
                                    "Sprints", "Sprint distance (m)", "Accels", "Decels", "Max speed"]
                        if c in pdata]
        st.dataframe(pdata[display_cols].round(2), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Pitch — needs tracking data, which a drill export does not contain
# ---------------------------------------------------------------------------

with tabs[IDX["Pitch"]]:
    st.markdown("#### Where it happened")
    st.caption(
        "These views need positional tracking — x/y for every player, many times a second. "
        "A StatSports drill export has totals and rates but no coordinates, so it cannot "
        "drive them. Load a tracking file, or use the open demo match."
    )

    source = st.radio(
        "Tracking source", ["Metrica open data (demo)", "Upload tracking CSV"],
        horizontal=True, key="pitch_source",
    )

    tracking = None
    events_df = None
    keeper = None

    if source.startswith("Metrica"):
        game = st.selectbox("Match", [1, 2], key="pitch_game",
                            format_func=lambda g: f"Metrica sample game {g}")
        side = st.radio("Side", ["Home", "Away"], horizontal=True, key="pitch_side")
        if st.button("Load tracking", key="pitch_load"):
            st.session_state.pitch_loaded = (game, side)

        loaded = st.session_state.get("pitch_loaded")
        if loaded:
            game, side = loaded
            with st.spinner("Downloading and processing tracking — takes a moment"):
                try:
                    tracking = cached_tracking(game, side)
                    events_df = cached_events(game)
                except TrackingFormatError as exc:
                    st.error(f"The open data didn't parse as tracking: {exc}")
                    tracking = None
                except Exception as exc:
                    st.error(
                        f"Couldn't fetch the open data: {type(exc).__name__} — {exc}. "
                        "This step needs outbound access to raw.githubusercontent.com."
                    )
                    tracking = None
        else:
            st.info("Press Load tracking to pull the match.")
    else:
        up = st.file_uploader("Tracking CSV (Metrica layout)", type=["csv"], key="pitch_up")
        st.caption(
            "Needs per-player x/y coordinates. A StatSports drill export will not work "
            "here — it has no positional data."
        )
        if up is not None:
            try:
                tracking = tidy_tracking(pd.read_csv(up, skiprows=2))
            except TrackingFormatError as exc:
                st.error(str(exc))
                tracking = None
            except Exception as exc:
                st.error(f"Couldn't read that file: {exc}")
                tracking = None

    candidates = []
    if tracking is not None and not tracking.empty:
        candidates = sorted({c[:-2] for c in tracking.columns if c.endswith("_x")
                             and not c.lower().startswith("ball")})
        if not candidates or "Period" not in tracking.columns:
            st.error(
                "That data loaded but doesn't have what the pitch views need: a Period "
                "column and one x/y pair per player."
            )
            with st.expander("What actually arrived"):
                st.write({
                    "rows": len(tracking),
                    "columns": len(tracking.columns),
                    "Period present": "Period" in tracking.columns,
                    "players detected": len(candidates),
                })
                st.code(", ".join(map(str, list(tracking.columns)[:24])) or "(no columns)")
                st.caption(
                    "If you are on the demo source and seeing this, the fetch returned "
                    "something other than the tracking file — usually a network block or "
                    "a changed URL. Clear the cache from the app menu and retry."
                )

    if candidates and "Period" in tracking.columns:
        # The keeper is whoever averages nearest his own goal across the match.
        auto = detect_keeper(tracking)
        keeper = st.selectbox(
            "Goalkeeper", candidates,
            index=candidates.index(auto) if auto in candidates else 0,
            key="pitch_keeper",
            help="Used to work out which way the team attacked in each half. It must be "
                 "someone who played the whole match, or one half can't be oriented.",
        )
        if auto and keeper == auto:
            st.caption(f"{auto} detected as the goalkeeper.")

        try:
            dirs = attacking_direction(tracking, keeper)
        except ValueError as exc:
            st.error(str(exc))
            dirs = None

        if dirs:
            st.caption(
                "Direction of play: "
                + ", ".join(f"{'H' + str(p)} {'left to right' if d == 1 else 'right to left'}"
                            for p, d in sorted(dirs.items()))
                + ". Second-half coordinates are flipped so the halves are comparable."
            )

    if candidates and "Period" in tracking.columns and dirs:
        positions = average_positions(tracking, keeper)
        phys = cached_physical(tracking, keeper)

        view = st.radio(
            "View",
            ["Average position by half", "Where the ball was won", "High-speed metres by zone"],
            key="pitch_view",
        )

        if view.startswith("Average"):
            metric = st.selectbox("Colour arrows by",
                                  ["HSR/min", "m/min", "Sprints"], key="pitch_metric")
            st.plotly_chart(
                shift_map(positions, phys, metric,
                          "Arrow = territorial change · colour = change in " + metric),
                use_container_width=True,
            )
            h1 = positions[positions["Period"] == "First half"]
            h2 = positions[positions["Period"] == "Second half"]
            if not h1.empty and not h2.empty:
                a = h1.nsmallest(4, "x")["x"].mean()
                b = h2.nsmallest(4, "x")["x"].mean()
                st.markdown(
                    f'<div class="flag flag-ok">Defensive line {a:.0f} m to {b:.0f} m '
                    f'({b - a:+.0f} m). Team average position {h1["x"].mean():.0f} m to '
                    f'{h2["x"].mean():.0f} m. Read this next to the running numbers: being '
                    "pushed back while working harder is a different problem from fading."
                    "</div>", unsafe_allow_html=True)

        elif view.startswith("Where"):
            if events_df is None:
                st.info("Recovery locations need the event file. Use the Metrica demo source.")
            else:
                team_side = st.session_state.get("pitch_loaded", (1, "Home"))[1]
                st.plotly_chart(
                    defensive_actions(events_df, team_side, dirs,
                                      title="Ball recoveries, with mean height marked"),
                    use_container_width=True,
                )
                st.caption("Press height measured off the pitch rather than asserted.")

        else:
            zones = cached_zones(tracking, keeper)
            st.plotly_chart(
                zone_difference(zones, title="High-speed metres, second half minus first"),
                use_container_width=True,
            )
            st.caption(
                "Green means more high-speed running in that zone in the second half. "
                "A falling total and a rising total in your own third are different findings."
            )


# ---------------------------------------------------------------------------
# TAB 4 — Data check
# ---------------------------------------------------------------------------

with tabs[IDX["Data check"]]:
    st.markdown("#### Before you report anything")
    st.caption("Run this every time. A clean-looking dashboard built on a broken export is worse than no dashboard.")

    for kind, msg in quality_flags(gps):
        cls = {"bad": "flag flag-bad", "warn": "flag", "ok": "flag flag-ok"}[kind]
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("#### Session reconciliation")
    st.caption("Whole-session totals against the sum of tagged drills. The gap is untagged time.")
    rec = reconcile(gps)
    if len(rec):
        st.dataframe(rec.round(1), use_container_width=True, hide_index=True)
    else:
        st.info("No whole-session row in this export, so nothing to reconcile.")

    st.markdown("#### Period mapping")
    st.caption("How each drill title was classified. Anything landing in 'Other' needs a rule adding.")
    mapping = gps[["Drill", "Period"]].drop_duplicates().sort_values("Period")
    st.dataframe(mapping, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Across matches
# ---------------------------------------------------------------------------

if len(matches) > 1:
    with tabs[IDX["Across matches"]]:
        totals = match_totals(library, min_exposure)
        per_period = match_period_rates(library, min_exposure)

        if totals.empty:
            st.warning("No match periods found across the library.")
        else:
            st.markdown("#### Squad trend")
            st.caption("Whole-match exposure-weighted rates, oldest to newest.")

            trend_metrics = [m for m in ("m/min", "HSR/min", "Mechanical load/min")
                             if m in totals]
            chosen = st.multiselect("Metrics", trend_metrics,
                                    default=trend_metrics[:2], key="trend_metrics")

            if chosen:
                short = [m.split(" v ")[0] if " v " not in m else m for m in totals["Match"]]
                fig = go.Figure()
                palette = [GRASS, AMBER, CORAL]
                for i, metric in enumerate(chosen):
                    fig.add_trace(go.Scatter(
                        x=totals["Match"], y=totals[metric], name=metric,
                        mode="lines+markers", line=dict(color=palette[i % 3], width=3),
                        marker=dict(size=9),
                        yaxis="y" if i == 0 else "y2",
                    ))
                layout = dict(PLOT_LAYOUT)
                layout.pop("yaxis")
                fig.update_layout(
                    height=380,
                    yaxis=dict(title=chosen[0], gridcolor=LINE),
                    yaxis2=dict(title=chosen[1] if len(chosen) > 1 else "",
                                overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)"),
                    **layout,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(totals.round(2), use_container_width=True, hide_index=True)

            st.markdown("#### First half against second half, by match")
            st.caption("Whether the within-match pattern repeats or varies with the opponent.")

            metric = st.selectbox("Metric", trend_metrics, key="hh_metric")
            pivot = per_period.pivot_table(index="Match", columns="Period", values=metric)
            pivot = pivot.reindex(index=matches,
                                  columns=[p for p in MATCH_PERIODS if p in pivot.columns])

            fig2 = go.Figure()
            colours = {"First half": GRASS, "Second half": AMBER, "Extra time": CORAL}
            for period in pivot.columns:
                fig2.add_bar(x=pivot.index, y=pivot[period], name=period,
                             marker_color=colours.get(period, MUTED))
            fig2.update_layout(barmode="group", height=360, **PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

            if len(pivot.columns) >= 2:
                drop = ((pivot.iloc[:, 1] - pivot.iloc[:, 0]) / pivot.iloc[:, 0] * 100)
                consistent = (drop < 0).all() or (drop > 0).all()
                direction = "falls" if drop.mean() < 0 else "rises"
                st.markdown(
                    f'<div class="flag flag-ok">{metric} {direction} from first to second half in '
                    f'{int((drop < 0).sum() if drop.mean() < 0 else (drop > 0).sum())} of '
                    f'{len(drop)} matches, by {abs(drop.mean()):.0f}% on average.'
                    + (" That is a pattern, not a one-off." if consistent else
                       " It varies by match, so it is worth looking at opponent and game state "
                       "before treating it as a conditioning issue.")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("#### Individual against own baseline")
            st.caption(
                f"Each player's most recent match against their own average from earlier matches. "
                f"A squad average hides the player who is well down on his own normal."
            )

            base_metric = st.selectbox("Baseline metric", trend_metrics, key="base_metric")
            baselines = player_baselines(library, base_metric, min_exposure)

            if baselines.empty:
                st.info("Not enough overlapping matches yet. Players need at least two to compare.")
            else:
                worst = baselines.head(3)
                cols = st.columns(min(3, len(worst)))
                for col, (_, row) in zip(cols, worst.iterrows()):
                    col.metric(row["Player"], f"{row['Latest']:.1f}",
                               f"{row['Change %']:+.1f}% vs own average")

                show = baselines.copy()
                if show["Z score"].isna().all():
                    show = show.drop(columns=["Z score"])
                    st.caption(
                        f"Z scores are withheld until a player has {Z_MIN_MATCHES} matches in "
                        "their baseline. A standard deviation from three matches is arithmetic, "
                        "not a finding."
                    )
                st.dataframe(show.round(2), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### Player by match")
            grid_metric = st.selectbox("Grid metric", trend_metrics, key="grid_metric")
            matrix = player_match_matrix(library, grid_metric, min_exposure)
            if matrix.empty:
                st.info("No players clear the minimum-minutes filter across these matches.")
            else:
                vals = matrix.to_numpy(dtype="float64")
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    styled = matrix.round(2).style.map(
                        shade(float(finite.min()), float(finite.max()))
                    ).format("{:.2f}", na_rep="-")
                else:
                    styled = matrix.round(2)
                st.dataframe(styled, use_container_width=True)
                st.markdown(
                    '<p class="note">Blank cells mean the player did not clear the minimum-minutes '
                    "filter in that match — usually an unused sub or a short cameo, not missing data.</p>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# TAB 5 — Export
# ---------------------------------------------------------------------------

with tabs[IDX["Export"]]:
    st.markdown("#### Match report (PDF)")
    st.caption("One page, built for a coach to read in thirty seconds. Includes the readings and the data-quality flags.")

    rc1, rc2 = st.columns(2)
    with rc1:
        report_title = st.text_input("Report title", value=f"Match report{' — ' + str(date_label) if date_label else ''}")
        squad_label = st.text_input("Squad / team", value="", placeholder="e.g. Ireland U19 men")
    with rc2:
        source_label = st.text_input("Data source", value="StatSports + match analytics")
        mark_sample = st.checkbox(
            "Mark as sample data", value=True,
            help="Stamps the footer. Leave on for anything that isn't a real club's match.",
        )

    if summary.empty:
        st.info("No match periods found, so there is nothing to report on yet.")
    else:
        try:
            pdf_bytes = build_match_report(
                summary=summary.round(2),
                joined=joined.round(3) if joined is not None else None,
                readings=readings,
                flags=quality_flags(gps),
                meta={
                    "title": report_title or "Match report",
                    "date": str(date_label) if date_label else "",
                    "squad": squad_label,
                    "source": source_label,
                    "sample": mark_sample,
                },
                squad=squad,
            )
            st.download_button(
                "Download match report (PDF)", pdf_bytes,
                file_name=f"match_report_{datetime.now():%Y%m%d}.pdf",
                mime="application/pdf", type="primary",
            )
        except Exception as exc:
            st.error(f"Couldn't build the PDF: {exc}")

    st.divider()
    st.markdown("#### Underlying data")

    clean_csv = gps.to_csv(index=False).encode()
    st.download_button("Cleaned GPS data (CSV)", clean_csv,
                       file_name=f"gps_clean_{datetime.now():%Y%m%d}.csv", mime="text/csv")

    if not summary.empty:
        st.download_button("Period summary (CSV)", summary.to_csv(index=False).encode(),
                           file_name=f"period_summary_{datetime.now():%Y%m%d}.csv", mime="text/csv")

    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            gps.to_excel(writer, sheet_name="GPS clean", index=False)
            if not summary.empty:
                summary.to_excel(writer, sheet_name="Period summary", index=False)
            reconcile(gps).to_excel(writer, sheet_name="Reconciliation", index=False)
        st.download_button("Full workbook (XLSX)", buf.getvalue(),
                           file_name=f"match_context_{datetime.now():%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        st.caption("Install openpyxl for the Excel workbook: pip install openpyxl")
