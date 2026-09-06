"""
report.py — one-page match report as a PDF.

Pure ReportLab. No kaleido, no wkhtmltopdf, no system binaries — so it installs
on Streamlit Community Cloud with nothing but `reportlab` in requirements.txt.

Charts are drawn with ReportLab's own graphics primitives rather than exported
from Plotly, which keeps the deployment dependency-free.
"""

import io
from datetime import datetime

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph

# Print palette — light, not the app's dark theme. Reports get printed.
INK = colors.HexColor("#14191D")
BODY = colors.HexColor("#3C4A52")
MUTED = colors.HexColor("#7C8C96")
RULE = colors.HexColor("#D6DEE2")
GRASS = colors.HexColor("#2E9E5B")
AMBER = colors.HexColor("#C4801E")
CORAL = colors.HexColor("#C2503F")
WASH = colors.HexColor("#F2F6F7")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

BODY_STYLE = ParagraphStyle(
    "body", fontName="Helvetica", fontSize=9, leading=12.5, textColor=BODY,
)
READING_STYLE = ParagraphStyle(
    "reading", fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK,
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
        self.c.setFillColor(INK)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= size + 4

    def caption(self, text, size=7.6):
        self.c.setFillColor(MUTED)
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
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(MARGIN, r.y, meta.get("title", "Match report"))

    c.setFillColor(GRASS)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(PAGE_W - MARGIN, r.y + 2, "MATCH CONTEXT")
    r.y -= 15

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)
    bits = [b for b in (meta.get("date"), meta.get("squad"), meta.get("source")) if b]
    c.drawString(MARGIN, r.y, "   |   ".join(bits))
    r.y -= 9
    r.rule(INK, 1.1)
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
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7)
        c.drawString(cx, r.y - 12, metric.upper())

        vals = []
        for p in periods:
            row = src[src["Period"] == p]
            vals.append(float(row[metric].iloc[0]) if len(row) and metric in row else float("nan"))

        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 13)
        txt = "  ".join(_fmt(v, 1 if metric != "PPDA" else 1) for v in vals)
        c.drawString(cx, r.y - 28, txt)

        if len(vals) >= 2:
            d = _delta(vals[0], vals[1])
            if d is not None:
                # For PPDA a fall means a more aggressive press, so green it.
                good_down = metric == "PPDA"
                colour = GRASS if ((d < 0) == good_down) else AMBER
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

    c.setFillColor(MUTED)
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
        c.setFillColor(INK)
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
                     fillColor=INK, textAnchor="middle"))
        d.add(String(cx, y - 10, lab, fontName="Helvetica", fontSize=7,
                     fillColor=MUTED, textAnchor="middle"))


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

    panels = [("Running rate  m/min", [float(v) for v in src["m/min"]], GRASS, "{:.0f}")]
    if "HSR/min" in src:
        panels.append(("High-speed running  m/min", [float(v) for v in src["HSR/min"]], AMBER, "{:.2f}"))
    if "Mechanical load/min" in src:
        panels.append(("Accel + decel  per min", [float(v) for v in src["Mechanical load/min"]], INK, "{:.2f}"))
    if has_press:
        panels.append(("Press intensity  100/PPDA", [100 / float(v) for v in src["PPDA"]], CORAL, "{:.1f}"))

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

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(MARGIN, r.y, "PLAYER")
    for i, col in enumerate(pivot.columns[1:]):
        label = {"First half": "H1", "Second half": "H2", "Extra time": "ET"}.get(col, col)
        c.drawRightString(MARGIN + name_w + (i + 1) * other_w - 6, r.y, label.upper())
    r.y -= 6
    r.rule()
    r.space(9)

    for _, row in pivot.iterrows():
        c.setFillColor(INK)
        c.setFont("Helvetica", 8.5)
        c.drawString(MARGIN, r.y, str(row["Player"])[:32])
        for i, col in enumerate(pivot.columns[1:]):
            v = row[col]
            x = MARGIN + name_w + (i + 1) * other_w - 6
            if col == "Change":
                if pd.isna(v):
                    c.setFillColor(MUTED)
                    c.setFont("Helvetica", 8.5)
                    c.drawRightString(x, r.y, "one period only")
                else:
                    c.setFillColor(GRASS if v >= 0 else CORAL)
                    c.setFont("Helvetica-Bold", 8.5)
                    c.drawRightString(x, r.y, f"{v:+.0f}%")
            else:
                c.setFillColor(INK)
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
        r.c.setStrokeColor(GRASS)
        r.c.setLineWidth(2)
        r.c.line(MARGIN + 1, top - 1, MARGIN + 1, r.y + 2)
        r.space(3)
    r.space(6)


def _flags(r: _Report, flags):
    if not flags:
        return
    r.heading("Data quality")
    for kind, msg in flags:
        colour = {"bad": CORAL, "warn": AMBER, "ok": GRASS}.get(kind, MUTED)
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
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    note = "Possession splits are apportioned, not measured."
    left = f"Generated {datetime.now():%d %b %Y, %H:%M}   |   page {page}"
    if not meta.get("sample"):
        left += f"   |   {note}"
    c.drawString(MARGIN, MARGIN + 7, left)
    if meta.get("sample"):
        c.setFillColor(CORAL)
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
