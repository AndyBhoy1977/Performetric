"""
sources.py — build the analytics half of the join from free, open event data.

StatsBomb publish event-level data for a set of competitions at
github.com/statsbomb/open-data. Event level is the point: xG, possession and
PPDA can all be computed *per half*, which is the granularity this app joins on.
Aggregator sites only publish whole-match figures, so they can't fill this gap.

Licence: the open data is free for public, non-commercial use under StatsBomb's
user agreement, and they ask that the source be credited in anything published.
Register at statsbomb.com/resource-centre and read the agreement before using it
in client work — for paid consultancy you need a licence, not this.

No API key, no scraping, no rate limits: the files are static JSON on GitHub.
"""

import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

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


def matches(competition_id: int, season_id: int) -> pd.DataFrame:
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


def metrica_tracking(game: int = 1, side: str = "Home") -> pd.DataFrame:
    """25 Hz tracking for one side, tidied into Period / Frame / Time / player x,y."""
    url = f"{METRICA}/Sample_Game_{game}/Sample_Game_{game}_RawTrackingData_{side}_Team.csv"
    raw = pd.read_csv(url, skiprows=2)

    # Columns arrive as Player11, Unnamed:4, Player1, Unnamed:6 ... — x then y.
    cols = list(raw.columns)
    renamed = {cols[0]: "Period", cols[1]: "Frame", cols[2]: "Time"}
    for i in range(3, len(cols) - 1, 2):
        name = str(cols[i])
        if name.startswith("Unnamed"):
            continue
        renamed[cols[i]] = f"{name}_x"
        renamed[cols[i + 1]] = f"{name}_y"
    raw = raw.rename(columns=renamed)
    keep = ["Period", "Frame", "Time"] + [c for c in raw.columns if c.endswith(("_x", "_y"))]
    return raw[keep]


def metrica_physical(game: int = 1, side: str = "Home",
                     smooth_frames: int = 7) -> pd.DataFrame:
    """Per-player, per-half physical metrics derived from tracking positions.

    Returns the same shape of metrics the GPS pipeline produces, so the output can
    be compared like for like: minutes, distance, m/min, HSR, sprints, accels.
    """
    track = metrica_tracking(game, side)
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
