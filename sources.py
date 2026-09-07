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
