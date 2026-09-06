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

from pipeline import (
    MATCH_PERIODS,
    WHOLE_SESSION,
    classify_period,
    load_gps as _load_gps,
    quality_flags,
    reconcile,
)


@st.cache_data(show_spinner=False)
def load_gps(file):
    return _load_gps(file)


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
gps_file = st.sidebar.file_uploader("StatSports export (CSV)", type=["csv"])

if gps_file is None:
    st.title("Match Context")
    st.markdown(
        '<p class="lede">Physical output read against what the team was actually doing on the pitch.</p>',
        unsafe_allow_html=True,
    )
    st.info("Upload a StatSports drill export to begin. Any export with player name, drill title and total distance will parse.")
    st.markdown(
        """
**What this does**

Your GPS export is split by drill — warm up, first half, second half, conditioning. Match analytics
are reported the same way: per half, per phase. This tool joins the two on that shared key and shows
physical output alongside the tactical picture that produced it.

**What it does not do**

It won't give you minute-by-minute synchronisation. That needs raw 10 Hz files and timestamped event
data, which is a different build.
        """
    )
    st.stop()

try:
    gps, colmap = load_gps(gps_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

periods_present = [p for p in MATCH_PERIODS if p in set(gps["Period"])]

st.sidebar.markdown("### Filters")
positions = sorted(gps["Position"].unique())
sel_positions = st.sidebar.multiselect("Position", positions, default=positions)

min_exposure = st.sidebar.slider("Minimum match minutes", 0, 60, 15, 5,
                                 help="Players below this are excluded from squad comparisons.")

gps = gps[gps["Position"].isin(sel_positions)]

# ---------------------------------------------------------------------------
# Analytics input — period level
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Match analytics")
st.sidebar.caption("Enter per period, or upload a CSV with a Period column.")

analytics_file = st.sidebar.file_uploader("Analytics CSV (optional)", type=["csv"], key="an")

default_analytics = pd.DataFrame(
    {
        "Period": periods_present or ["First half", "Second half"],
        "xG for": [0.0] * max(len(periods_present), 2),
        "xG against": [0.0] * max(len(periods_present), 2),
        "PPDA": [10.0] * max(len(periods_present), 2),
        "Opp PPDA": [10.0] * max(len(periods_present), 2),
        "Possession %": [50.0] * max(len(periods_present), 2),
        "Field tilt %": [50.0] * max(len(periods_present), 2),
    }
)

if analytics_file is not None:
    analytics = pd.read_csv(analytics_file)
    analytics.columns = [c.strip() for c in analytics.columns]
else:
    analytics = default_analytics

st.title("Match Context")
date_label = str(gps["Date"].iloc[0]) if "Date" in gps and len(gps) else ""
st.markdown(
    f'<p class="lede">StatSports export joined to match analytics at period level'
    + (f" · {date_label}" if date_label else "")
    + "</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Period profile", "Physical in context", "Players", "Data check", "Export"])

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

with tabs[0]:
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

with tabs[1]:
    st.markdown("#### Match analytics")
    st.caption("Edit these to match your Opta / Wyscout / StatsBomb period splits. Values persist while the app runs.")

    edited = st.data_editor(
        analytics,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="analytics_editor",
    )

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
            joined["HSR per unit of press"] = joined["HSR/min"] / joined["PPDA"]
            joined["xG per 1000 m"] = joined["xG for"] / (joined["Distance (m)"] / 1000)
            joined["Mechanical cost of press"] = joined["Mechanical load/min"] / joined["PPDA"]

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

                d_rate = (b["m/min"] - a["m/min"]) / a["m/min"] * 100 if a["m/min"] else 0
                d_ppda = (b["PPDA"] - a["PPDA"]) / a["PPDA"] * 100 if a["PPDA"] else 0

                if d_rate < -5 and d_ppda > 5:
                    readings.append(
                        f"Running rate fell {abs(d_rate):.0f}% and the press loosened ({d_ppda:+.0f}% PPDA). "
                        "Consistent picture — the team dropped off and the physical numbers follow. "
                        "Whether that was chosen or forced is a coaching conversation, not a data one."
                    )
                elif d_rate < -5 and d_ppda < -5:
                    readings.append(
                        f"Running rate fell {abs(d_rate):.0f}% while the press intensified ({d_ppda:.0f}% PPDA). "
                        "The team asked for more pressure from less output. Worth checking whether the press was "
                        "being triggered but not supported."
                    )
                elif d_rate > 5 and d_ppda > 5:
                    readings.append(
                        f"Running rate rose {d_rate:.0f}% while the press loosened. Output went into recovery runs "
                        "and transitions rather than pressure — often a chasing-the-game profile."
                    )
                else:
                    readings.append(
                        f"Running rate moved {d_rate:+.0f}% and PPDA {d_ppda:+.0f}%. Both broadly stable across halves."
                    )

                if "Mechanical cost of press" in joined:
                    if b["Mechanical cost of press"] > a["Mechanical cost of press"] * 1.15:
                        readings.append(
                            "Accel/decel cost per unit of press rose in the second half — the same pressure is "
                            "being bought with more braking. That is the load that shows up in the following days."
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

with tabs[2]:
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

with tabs[3]:
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
# TAB 5 — Export
# ---------------------------------------------------------------------------

with tabs[4]:
    st.markdown("#### Take it away")

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
