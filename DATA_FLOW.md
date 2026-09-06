# Data flow

The repeatable process, as built. Two files: `pipeline.py` does the work, `app.py` displays it.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload the StatSports CSV in the sidebar. Enter the analytics figures in the second tab.

---

## The join key is the period, not the minute

This is the one architectural decision everything else follows from.

A StatSports drill export gives one row per player per drill. Your file has five drill types:
warm up, first half, second half, post-match conditioning, and a whole-session container row.
There is no minute column, and no timestamped events — so minute-by-minute synchronisation
isn't available from this export.

Match analytics are reported the same way: xG, PPDA and possession are almost always given per
half or per phase. So both sides of the join already share a natural key. Use it.

```
StatSports drill export (CSV)
        │
        ├── map_columns()        tolerant header matching, handles naming variants
        ├── parse_duration()     "00:47:17" → 47.28 minutes
        ├── classify_period()    "3. Second Half v Switzerland" → "Second half"
        └── derive rates         m/min, HSR/min, mechanical load/min
        │
        ▼
   clean player × period table
        │
        ├── filter: match periods only, drop the whole-session container row
        ├── filter: minimum exposure threshold
        └── period_summary()     exposure-weighted squad rates
        │
        ▼
   squad × period summary ──── join on Period ──── analytics (xG, PPDA, possession)
        │
        ▼
   contextual measures → readings → export
```

## Two traps this pipeline avoids

**Double counting.** The `Entire Session - Live` row is a container, not a drill. Summing every
row inflates every total by roughly 2×. It's classified separately and excluded from all
period aggregation — then used deliberately in the reconciliation check.

**Totals across unequal exposure.** In your file, second-half minutes range from 10:45 to 50:29.
Ranking players on total distance would just be ranking them on minutes played. Every squad
comparison uses rates, and totals are divided by summed minutes rather than averaged across
players — averaging a per-minute column would weight an 11-minute cameo the same as a full half.

---

## What the pipeline found in your file

Running it on `Dummy.csv`, unprompted:

| Check | Result |
|---|---|
| Heart rate | Unusable. Max HR is 80 bpm for every player — below plausible match values. All HR metrics should be treated as missing. |
| HR dropout | Finn Brennan, Marco Silva, Tadhg Kelly show average HR under 60 with a max of 80. |
| Low exposure | Cian Murphy and Noah Williams have ~11 match minutes. Rates only. |
| No match period | Liam O'Brien and Sean Doyle have a warm-up and a session row but no half — unused subs. |
| Unallocated distance | Six players have >25% of session distance outside tagged drills. Sean Doyle 51%, Liam O'Brien 50%, Noah Williams 44%. Consistent with unused subs doing separate work. |

That QA panel is worth as much to a club as the charts. Most integration work fails on data
quality, not analysis, and a dashboard that renders confidently over a broken export is worse
than no dashboard.

---

## Derived measures

| Measure | How | Read it as |
|---|---|---|
| m/min | distance ÷ minutes | Baseline running rate |
| HSR/min | high-speed running ÷ minutes | Intensity, not volume |
| Mechanical load/min | (accels + decels) ÷ minutes | The braking cost that shows up on MD+1 |
| In / out of possession m/min | m/min apportioned by possession share | Directional only — see caveat |
| HSR per unit of press | HSR/min ÷ PPDA | Running being spent on pressure |
| Mechanical cost of press | mechanical load/min ÷ PPDA | Whether the press is getting more expensive |
| xG per 1000 m | xG ÷ (distance ÷ 1000) | Output per unit of work |

**The possession caveat matters.** Splitting distance by possession share is apportionment, not
measurement. It's defensible at period level and useful for a conversation, but it is not the
same as knowing where a player ran while the team had the ball. Say so to clients — the
credibility you get from naming that limit is worth more than the extra chart.

---

## Adding periods

`classify_period()` maps drill titles by regex. Every club names drills differently. To add a rule:

```python
PERIOD_RULES = [
    ...
    (r"press(ing)?\s*block|rondo", "Possession block"),
]
```

The Data check tab lists every drill title and what it mapped to. Anything landing in `Other`
needs a rule. That tab is the first thing to open with a new club's export.

---

## Analytics input

The app takes analytics two ways: an editable table (fastest for a one-off report) or a CSV with
a `Period` column matching the GPS period labels.

```csv
Period,xG for,xG against,PPDA,Opp PPDA,Possession %,Field tilt %
First half,0.9,0.3,8.4,11.0,58,61
Second half,0.4,1.1,12.9,9.2,44,39
```

Pull these from Opta, Wyscout, StatsBomb or InStat by filtering events to each half and
aggregating. PPDA is opposition passes divided by your defensive actions in the pressing zone —
check which definition your provider uses before comparing across sources, because they differ.

---

## Going further

Ranked by value per unit of effort:

1. **Multiple matches.** One match is anecdote. The same pipeline over a season turns these into
   trends, and trends are what clubs pay for. Add a match identifier and stack the exports.
2. **Position groups.** Your file has position already. m/min means something different for a
   defender and a midfielder, and squad averages hide that.
3. **Individual baselines.** Compare a player to his own rolling average, not the squad's. This is
   where the injury-risk conversation actually starts.
4. **Measured possession splits.** Needs event data with timestamps. Real work, real value — the
   thing that would genuinely differentiate the service.

Do 1 and 2 before adding anything else. They cost little and change the conversation from
"here is a match" to "here is a pattern".
