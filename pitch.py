"""
pitch.py — pitch visualisations that put physical output in tactical context.

The aim is to replace "we faded in the second half" with a picture of what
actually changed: where the team stood, where they won the ball, and whether
the running went with the territory or against it.

Plotly only, so it renders inside the app with no new dependency.

THE ONE THING THAT WILL RUIN THIS IF YOU GET IT WRONG
-----------------------------------------------------
Teams switch ends at half time. In the Metrica sample the home keeper averages
x=0.13 in the first half and x=0.91 in the second. Compare halves on raw
coordinates and every player appears to have advanced 60 metres. Every function
here normalises so the team always attacks left to right, in both halves.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PITCH_X, PITCH_Y = 105.0, 68.0

INK = "#12181C"
LINE = "#3A4750"
CHALK = "#C9D6DC"
MUTED = "#8FA3AD"
GRASS = "#4FB477"
AMBER = "#E8A33D"
CORAL = "#E06C5E"


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def draw_pitch(height=560, title=""):
    """An empty pitch, 105 x 68, origin bottom left, attacking left to right."""
    fig = go.Figure()
    shapes = []

    def rect(x0, y0, x1, y1):
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=LINE, width=1.4), layer="below"))

    rect(0, 0, PITCH_X, PITCH_Y)                       # touchlines
    rect(0, 13.84, 16.5, 54.16)                        # left penalty area
    rect(PITCH_X - 16.5, 13.84, PITCH_X, 54.16)        # right penalty area
    rect(0, 24.84, 5.5, 43.16)                         # left six yard
    rect(PITCH_X - 5.5, 24.84, PITCH_X, 43.16)         # right six yard

    shapes.append(dict(type="line", x0=PITCH_X / 2, y0=0, x1=PITCH_X / 2, y1=PITCH_Y,
                       line=dict(color=LINE, width=1.4), layer="below"))
    shapes.append(dict(type="circle", x0=PITCH_X / 2 - 9.15, y0=PITCH_Y / 2 - 9.15,
                       x1=PITCH_X / 2 + 9.15, y1=PITCH_Y / 2 + 9.15,
                       line=dict(color=LINE, width=1.4), layer="below"))

    fig.update_layout(
        shapes=shapes,
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHALK, size=12),
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
                       arrowcolor=MUTED, text="")
    fig.add_annotation(x=PITCH_X / 2 + 14, y=-2.2, text="attacking", showarrow=False,
                       font=dict(color=MUTED, size=10))
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
        marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=MUTED, width=1.6)),
        name="First half", hovertext=shared, hoverinfo="text",
    ))

    # Arrows: first-half position to second-half position
    for player in shared:
        delta = change.get(player)
        colour = MUTED if delta is None else (GRASS if delta >= 0 else CORAL)
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
                    color=[GRASS if change.get(p, 0) >= 0 else CORAL for p in shared],
                    opacity=0.9, line=dict(color=INK, width=1.5)),
        text=labels, textposition="middle center",
        textfont=dict(color=INK, size=10, family="Arial Black"),
        name="Second half", hovertext=hover, hoverinfo="text",
    ))

    # Defensive line height in each half. Labels are staggered vertically because
    # the two lines sit close together whenever the shift is small — which is
    # exactly when you most need to read both numbers.
    for i, (frame, colour, label) in enumerate(
            ((h1.loc[shared], MUTED, "H1"), (h2.loc[shared], CHALK, "H2"))):
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

    palette = {"First half": AMBER, "Second half": GRASS}
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
                        line=dict(color=INK, width=1)),
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
            colour = GRASS if t >= 0 else CORAL
            fig.add_shape(type="rect", layer="below",
                          x0=ix * cell_w, y0=iy * cell_h,
                          x1=(ix + 1) * cell_w, y1=(iy + 1) * cell_h,
                          line=dict(width=0),
                          fillcolor=colour, opacity=min(abs(t) * 0.6, 0.6))
            fig.add_annotation(x=(ix + 0.5) * cell_w, y=(iy + 0.5) * cell_h,
                               text=f"{v:+.0f}", showarrow=False,
                               font=dict(color=CHALK, size=9))
    return fig
