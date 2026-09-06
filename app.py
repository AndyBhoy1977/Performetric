"""
Match Context — StatSports GPS x Match Analytics
Period-level integration of physical output and tactical metrics.
"""

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

from report import build_match_report
from pipeline import (
    MATCH_PERIODS,
    WHOLE_SESSION,
    Z_MIN_MATCHES,
    classify_period,
    load_gps as _load_gps,
    load_many as _load_many,
    match_period_rates,
    match_totals,
    ordered_matches,
    player_baselines,
    player_match_matrix,
    quality_flags,
    reconcile,
)


@st.cache_data(show_spinner=False)
def load_gps(file):
    return _load_gps(file)


@st.cache_data(show_spinner=False)
def load_library(payloads):
    """payloads: tuple of (name, bytes). Hashable, so Streamlit can cache it."""
    return _load_many([(name, io.BytesIO(data)) for name, data in payloads])


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

tab_names = ["Period profile", "Physical in context", "Players", "Data check", "Export"]
if len(matches) > 1:
    tab_names.insert(3, "Across matches")
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
                st.dataframe(
                    matrix.round(2).style.background_gradient(cmap="RdYlGn", axis=None),
                    use_container_width=True,
                )
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
