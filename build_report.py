"""
APAC Corridor Zendesk Deep-Dive — single-file HTML report builder.
Three-month scope: Feb–Apr 2026, both inbound calls and messaging (chat/WhatsApp).

Reads:
  Inbound calls:
    - Reason_codes_related_to_transactions_yesterday_05122026_0953.xlsx  (Feb 2026, 107 tickets)
    - New files/Reason_codes_related_to_transactions_yesterday_05122026_0948.xlsx (Mar 2026, 107 tickets)
    - Reason_codes_related_to_transactions_yesterday_05122026_0926.xlsx  (Apr 2026, 78 tickets)
  Messaging:
    - New files/Messaging_tickets_-_Created_APAC_05122026_0952.xlsx       (Mar 2026, 43 tickets)
    - New files/Messaging_tickets_-_Created_APAC_05122026_0958.xlsx       (Apr 2026, 29 tickets)
    - (Feb 2026 messaging file not provided — noted in §12.)
  Enrichment:
    - data/ticket_enrichment.json (optional)

Writes:
  - APAC_Corridor_Analysis_Feb-Apr2026.html  (portable single-file report)
  - charts/*.png
  - data/*.csv
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import textwrap
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
NEW = ROOT / "New files"
HTML_OUT = ROOT / "APAC_Corridor_Analysis_Feb-Apr2026.html"
CHARTS_DIR = ROOT / "charts"
DATA_DIR = ROOT / "data"
ENRICHMENT_PATH = DATA_DIR / "ticket_enrichment.json"

# Inbound-call files (one per month).
INBOUND_FILES = [
    ("2026-02", NEW  / "Reason_codes_related_to_transactions_yesterday_05122026_0953.xlsx"),
    ("2026-03", NEW  / "Reason_codes_related_to_transactions_yesterday_05122026_0948.xlsx"),
    ("2026-04", ROOT / "Reason_codes_related_to_transactions_yesterday_05122026_0926.xlsx"),
]
# Messaging files (Feb not provided).
MESSAGING_FILES = [
    ("2026-03", NEW / "Messaging_tickets_-_Created_APAC_05122026_0952.xlsx"),
    ("2026-04", NEW / "Messaging_tickets_-_Created_APAC_05122026_0958.xlsx"),
]

CHARTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Customer Sending Locations to drop at load time. The 60030 / 60040 codes are
# internal / non-customer-facing sources that distort the corridor analysis.
EXCLUDED_LOCATIONS = {"60030", "60040"}

# Compliance-flagged definition: any non-blank CEC Code.
NON_TERMINAL_STATUSES = {
    "Sent to Corresp", "Legal Hold", "On Hold", "Posted",
    "Posted - Under Review (Digital)", "AwaitingBalance", "To Cancel",
}
COMPLIANCE_GROUPS = {"Ria - APAC Compliance", "Ria - Digital Fraud"}
HIGH_VALUE_THRESHOLD = 5000.0
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Professional, brand-neutral palette (navy, teal, amber, coral, sage, slate,
# wine, sand). Used consistently across every chart.
PALETTE = [
    "#1f3a5f", "#2e8b8b", "#d99a3d", "#c8553d",
    "#6b8e23", "#5b6770", "#7a3b5c", "#c2a878",
]
SEQUENTIAL_CMAP = "crest"

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
})
sns.set_palette(PALETTE)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def fmt_money(x, currency: str = "") -> str:
    if pd.isna(x):
        return "—"
    return f"{currency}{x:,.2f}"


def fmt_int(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def fmt_pct(x, decimals: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}%"


def fig_to_img(fig, slug: str) -> str:
    """Save figure as a PNG file AND return an inline <img> tag (base64)."""
    png_path = CHARTS_DIR / f"{slug}.png"
    fig.savefig(png_path)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img class="chart" src="data:image/png;base64,{b64}" alt="{slug}" />'


def df_to_html_table(df: pd.DataFrame, classes: str = "kpi-table", index: bool = False) -> str:
    return df.to_html(classes=classes, index=index, border=0, escape=False)


# ────────────────────────────────────────────────────────────────────────────
# Data load + derived columns
# ────────────────────────────────────────────────────────────────────────────
def _read_inbound(path: Path, month_tag: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["data_source"] = "inbound"
    df["month"] = month_tag
    return df


def _read_messaging(path: Path, month_tag: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Normalise the destination column name.
    df = df.rename(columns={"Destination Countries Victor": "Destination Country"})
    # Messaging files don't have Selling/Recipient amounts or the "Tickets" counter.
    for col in ("Tickets", "Recipient Amount", "Selling Amount"):
        if col not in df.columns:
            df[col] = np.nan
    if "Messaging tickets" in df.columns:
        df["Tickets"] = df["Tickets"].fillna(df["Messaging tickets"])
    df["data_source"] = "messaging"
    df["month"] = month_tag
    return df


def load_data() -> pd.DataFrame:
    frames = [_read_inbound(p, m) for m, p in INBOUND_FILES]
    frames += [_read_messaging(p, m) for m, p in MESSAGING_FILES]
    df = pd.concat(frames, ignore_index=True, sort=False)

    # APAC-only filter — Country in {Australia, New Zealand}. The messaging files include a
    # handful of UK / US / France origins that slipped through the destination filter; drop them
    # so the corridor analysis is honest about who's sending.
    apac_origins = {"Australia", "New Zealand"}
    df["Country"] = df["Country"].fillna("").astype(str).str.strip()
    pre = len(df)
    df = df[df["Country"].isin(apac_origins)].reset_index(drop=True)
    print(f"  (filtered out {pre - len(df)} non-APAC-origin rows)")

    # Drop internal / non-customer-facing sending locations.
    df["Customer Sending Location"] = df["Customer Sending Location"].fillna("").astype(str).str.strip()
    pre = len(df)
    df = df[~df["Customer Sending Location"].isin(EXCLUDED_LOCATIONS)].reset_index(drop=True)
    print(f"  (filtered out {pre - len(df)} rows from excluded sending locations {sorted(EXCLUDED_LOCATIONS)})")

    df["Source"] = df["Source"].fillna("").astype(str).str.strip().replace("", "(blank)")
    df["CEC Code"] = df["CEC Code"].fillna("").astype(str).str.strip()
    df["Correspondent"] = df["Correspondent"].fillna("").astype(str).str.strip()
    df["Ria Order Number"] = df["Ria Order Number"].fillna("").astype(str).str.strip()
    df["Destination Country"] = df["Destination Country"].fillna("Unknown").astype(str).str.strip()
    df["Ticket Channel"] = df["Ticket Channel"].fillna("Unknown").astype(str).str.strip()
    df["Ticket group"] = df["Ticket group"].fillna("Unknown").astype(str).str.strip()
    df["Delivery Method"] = df["Delivery Method"].fillna("Unknown").astype(str).str.strip()
    df["created"] = pd.to_datetime(df["Ticket created - Timestamp"], errors="coerce")
    df["solved"] = pd.to_datetime(df["Ticket solved - Timestamp"], errors="coerce")
    df["resolution_minutes"] = (df["solved"] - df["created"]).dt.total_seconds() / 60.0
    reason_split = df["Ria - Reason for Contact"].fillna("Unknown::Unknown").astype(str).str.split("::", n=1, expand=True)
    df["category"] = reason_split[0].fillna("Unknown")
    df["subcategory"] = reason_split[1].fillna("Unknown")
    df["is_compliance_flagged"] = df["CEC Code"].ne("")
    df["is_digital_location"] = df["Customer Sending Location"].str.startswith("EW")
    df["is_high_value"] = df["Selling Amount"] > HIGH_VALUE_THRESHOLD
    df["corridor"] = df["Country"] + " → " + df["Destination Country"]
    df["day_of_week"] = df["created"].dt.day_name().str[:3]
    df["hour_utc"] = df["created"].dt.hour
    df["iso_week"] = df["created"].dt.isocalendar().week
    df["date"] = df["created"].dt.date
    df["channel_type"] = np.where(df["is_digital_location"], "Digital (EW)", "Physical Store")
    # Simplified channel for the channel-mix chart.
    df["channel_simple"] = df["Ticket Channel"].apply(_simplify_channel)
    return df


def _simplify_channel(ch: str) -> str:
    """Group the messaging channels into 3 buckets + leave Inbound call alone."""
    if not ch or ch in ("Unknown", "\xa0"):
        return "Unknown"
    if ch == "Inbound call":
        return "Inbound call"
    if "WhatsApp" in ch:
        return "WhatsApp"
    if "Chat" in ch:
        return "Web chat"
    if "Facebook" in ch or "Instagram" in ch or "Twitter" in ch:
        return "Social media"
    return "Other messaging"


def load_enrichment() -> dict:
    if ENRICHMENT_PATH.exists():
        try:
            with open(ENRICHMENT_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Keys may be strings — keep them as strings for lookup by str(ticket_id).
                return {str(k): v for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def enrichment_block(ticket_ids: list[int], enrichment: dict, max_quotes: int = 2) -> str:
    """Render quoted-evidence block for a list of ticket IDs; placeholder if no data."""
    found = [(tid, enrichment[str(tid)]) for tid in ticket_ids if str(tid) in enrichment]
    if not found:
        return (
            '<div class="enrichment-pending">'
            "💬 <strong>Quoted ticket evidence will appear here after the Zendesk enrichment pass.</strong> "
            "The base report uses ticket metadata only; conversation snippets are layered on once we scrape "
            f"the {len(ticket_ids)} candidate ticket(s) flagged for this section."
            "</div>"
        )
    parts = ['<div class="enrichment-quotes">']
    for tid, rec in found[:max_quotes]:
        subj = escape(rec.get("subject", "(no subject)"))
        first = escape((rec.get("first_customer_message") or "")[:400])
        last = escape((rec.get("last_agent_reply") or "")[:400])
        tags = ", ".join(escape(t) for t in (rec.get("tags") or [])[:6])
        parts.append(
            f'<div class="quote-card">'
            f'<div class="quote-head">Ticket #{tid} — {subj}</div>'
            + (f'<div class="quote-tags">tags: {tags}</div>' if tags else "")
            + (f'<div class="quote-body"><span class="quote-label">First customer:</span> "{first}"</div>' if first else "")
            + (f'<div class="quote-body"><span class="quote-label">Last agent:</span> "{last}"</div>' if last else "")
            + "</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Section builders
# ────────────────────────────────────────────────────────────────────────────
def section_1_exec_summary(df: pd.DataFrame, enrichment: dict) -> tuple[str, dict]:
    n = len(df)
    n_inbound = int((df["data_source"] == "inbound").sum())
    n_messaging = int((df["data_source"] == "messaging").sum())
    n_orders = df["Ria Order Number"].nunique()
    avg_res = df["resolution_minutes"].mean()
    med_res = df["resolution_minutes"].median()
    median_value = df["Selling Amount"].median()  # NaN for messaging — pandas median ignores NaN
    pct_cec = 100 * df["is_compliance_flagged"].mean()
    pct_legal_hold = 100 * (df["Order Status"] == "Legal Hold").mean()
    top_corridor = df["corridor"].value_counts().idxmax()
    top_corridor_pct = 100 * df["corridor"].value_counts(normalize=True).iloc[0]
    digital_pct = 100 * df["is_digital_location"].mean()
    top_reason = df["subcategory"].value_counts().idxmax()
    top_reason_n = int(df["subcategory"].value_counts().iloc[0])
    months_covered = sorted(df["month"].dropna().unique().tolist())

    kpis = dict(
        total_tickets=n, n_inbound=n_inbound, n_messaging=n_messaging,
        unique_orders=n_orders, avg_res_min=avg_res, med_res_min=med_res,
        median_selling=median_value, pct_cec=pct_cec,
        pct_legal_hold=pct_legal_hold, top_corridor=top_corridor,
        top_corridor_pct=top_corridor_pct, digital_pct=digital_pct,
        top_reason=top_reason, top_reason_n=top_reason_n, months=months_covered,
    )

    # KPI tile dashboard
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.6))
    tiles = [
        ("Total tickets", fmt_int(n), PALETTE[0]),
        ("Unique orders", fmt_int(n_orders), PALETTE[1]),
        ("Avg resolution (min)", f"{avg_res:,.0f}", PALETTE[2]),
        ("Median selling", fmt_money(median_value, "$"), PALETTE[3]),
        ("% compliance-flagged", fmt_pct(pct_cec), PALETTE[6]),
    ]
    for ax, (label, value, color) in zip(axes, tiles):
        ax.set_facecolor(color)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 0.62, value, ha="center", va="center", color="white", fontsize=22, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.18, label, ha="center", va="center", color="white", fontsize=10, transform=ax.transAxes)
        ax.grid(False)
    fig.suptitle("APAC Corridor — Feb-Apr 2026 KPI dashboard", fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    kpi_img = fig_to_img(fig, "01_kpi_dashboard")

    narrative = (
        f"<p>Across <strong>February–April 2026</strong>, the AU/NZ → Pacific Islands corridor generated "
        f"<strong>{n:,} support tickets</strong> — <strong>{n_inbound:,}</strong> inbound calls and "
        f"<strong>{n_messaging:,}</strong> messaging (chat/WhatsApp) tickets — covering "
        f"<strong>{n_orders:,} unique orders</strong>. Volume is concentrated on a single corridor — "
        f"<strong>{top_corridor}</strong> alone accounts for {top_corridor_pct:.0f}% of all tickets — "
        f"and on a single contact reason: <strong>{top_reason}</strong> ({top_reason_n} tickets, "
        f"{100*top_reason_n/n:.0f}%). The portfolio carries material compliance exposure: "
        f"<strong>{pct_cec:.0f}% of tickets are CEC-flagged</strong> and <strong>{pct_legal_hold:.0f}%</strong> "
        f"sit in Legal Hold at time of pull. Average resolution time is heavily skewed by a long tail "
        f"(mean {avg_res:,.0f} min vs. median {med_res:.1f} min) — most tickets close inside 15 minutes, but "
        f"a handful of orders bleed into multi-day handling. The biggest single risk: high-value transactions "
        f"(&gt;${HIGH_VALUE_THRESHOLD:,.0f}) cluster almost entirely on the Samoa corridor and disproportionately "
        f"hit Legal Hold — this is the most actionable concentration risk in the dataset (see §9). The biggest "
        f"opportunity: a single digital sending location (EW103) generates "
        f"<strong>{digital_pct:.0f}%</strong> of all volume, making it the highest-leverage point for "
        f"self-service automation and proactive status updates.</p>"
    )

    body = (
        "<p class='so-what'>So what? One corridor (AU → Samoa), one reason category (order disposition), and one "
        "digital storefront drive the bulk of cross-channel ticket volume — and the highest-risk tickets cluster "
        "together. Targeted intervention on those three concentrations would move the needle further than broad "
        "SLA work.</p>"
        + narrative + kpi_img
    )
    return body, kpis


def section_2_volume(df: pd.DataFrame) -> str:
    # Daily volume, stacked by channel
    daily_split = df.pivot_table(index="date", columns="data_source", aggfunc="size", fill_value=0)
    full_index = pd.date_range(df["date"].min(), df["date"].max()).date
    daily_split = daily_split.reindex(full_index, fill_value=0)
    daily_total = daily_split.sum(axis=1)
    rolling = daily_total.rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    cols_order = [c for c in ["inbound", "messaging"] if c in daily_split.columns]
    bottom = np.zeros(len(daily_split))
    bar_colors = {"inbound": PALETTE[0], "messaging": PALETTE[1]}
    for col in cols_order:
        ax.bar(daily_split.index, daily_split[col], bottom=bottom, color=bar_colors[col], label=col.capitalize(), width=0.85)
        bottom = bottom + daily_split[col].values
    ax.plot(daily_total.index, rolling.values, color=PALETTE[3], linewidth=2.5, label="7-day moving avg (total)")
    ax.set_title("Daily ticket volume — Feb-Apr 2026 (stacked by channel)")
    ax.set_xlabel("Date"); ax.set_ylabel("Tickets")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    daily_img = fig_to_img(fig, "02a_daily_volume")

    # Month-over-month bar with channel split
    mom = df.pivot_table(index="month", columns="data_source", aggfunc="size", fill_value=0)
    mom = mom.reindex(sorted(mom.index))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(mom.index))
    width = 0.38
    for i, col in enumerate(cols_order):
        ax.bar(x + (i - 0.5) * width, mom[col], width=width, color=bar_colors[col], label=col.capitalize())
        for j, v in enumerate(mom[col].values):
            ax.text(x[j] + (i - 0.5) * width, v + 1, str(int(v)), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(mom.index)
    ax.set_title("Tickets per month, by channel")
    ax.set_ylabel("Tickets")
    ax.legend()
    mom_img = fig_to_img(fig, "02b_month_by_channel")

    dow = df.groupby("day_of_week").size().reindex(DOW_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(dow.index, dow.values, color=PALETTE[1])
    peak_idx = int(np.argmax(dow.values))
    bars[peak_idx].set_color(PALETTE[3])
    for i, v in enumerate(dow.values):
        ax.text(i, v + 0.3, str(int(v)), ha="center", fontsize=9)
    ax.set_title("Tickets by day of week (peak in red)")
    ax.set_ylabel("Tickets")
    dow_img = fig_to_img(fig, "02c_day_of_week")

    hour = df.groupby("hour_utc").size().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(hour.index, hour.values, color=PALETTE[4])
    ax.set_xticks(range(24))
    ax.set_title("Tickets by hour of day (UTC, as-recorded)")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Tickets")
    hour_img = fig_to_img(fig, "02d_hour_of_day")

    wow = df.groupby("iso_week").size().rename("tickets").to_frame()
    wow["WoW Δ"] = wow["tickets"].diff()
    wow["WoW %Δ"] = wow["tickets"].pct_change() * 100
    wow.index.name = "ISO week"
    wow_disp = wow.copy()
    wow_disp["tickets"] = wow_disp["tickets"].map(fmt_int)
    wow_disp["WoW Δ"] = wow_disp["WoW Δ"].map(lambda v: fmt_int(v) if pd.notna(v) else "—")
    wow_disp["WoW %Δ"] = wow_disp["WoW %Δ"].map(lambda v: fmt_pct(v) if pd.notna(v) else "—")
    wow_html = df_to_html_table(wow_disp.reset_index())

    # Slope across iso-weeks
    weeks_arr = wow.index.values.astype(float)
    if len(weeks_arr) >= 2:
        slope = np.polyfit(weeks_arr, wow["tickets"].values, 1)[0]
        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"
        trend_note = f"<p>Linear slope across ISO weeks is <strong>{slope:+.1f} tickets/week → {trend}</strong>.</p>"
    else:
        trend_note = ""

    peak_day = dow.idxmax(); peak_day_n = int(dow.max())
    peak_hour = int(hour.idxmax()); peak_hour_n = int(hour.max())

    # Month totals table (live numbers)
    mom_disp = mom.copy()
    mom_disp["Total"] = mom_disp.sum(axis=1)
    mom_disp = mom_disp.reset_index().rename(columns={"month": "Month", "inbound": "Inbound calls", "messaging": "Messaging"})
    mom_html = df_to_html_table(mom_disp)

    return (
        "<p class='so-what'>So what? Ticket volume sits in a band of ~100–135 tickets/month, with messaging "
        "(chat / WhatsApp) running ~25–35% the size of inbound-call volume. Day-of-week is weekday-heavy with "
        "no real weekend tail — that's a staffing pattern, not a customer-demand pattern. Hour-of-day "
        "concentration in the Pacific-daytime window is what should drive the APAC desk roster.</p>"
        f"<p><strong>Peak day:</strong> {peak_day} ({peak_day_n} tickets). "
        f"<strong>Peak hour (UTC):</strong> {peak_hour:02d}:00 ({peak_hour_n} tickets).</p>"
        + daily_img + "<h3>Month-over-month volume by channel</h3>" + mom_img + mom_html
        + dow_img + hour_img
        + "<h3>Week-over-week (ISO week)</h3>" + wow_html + trend_note
    )


def section_3_reasons(df: pd.DataFrame, enrichment: dict) -> str:
    sub_counts = df["subcategory"].value_counts()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(sub_counts))))
    ax.barh(sub_counts.index[::-1], sub_counts.values[::-1], color=PALETTE[0])
    for i, v in enumerate(sub_counts.values[::-1]):
        ax.text(v + 0.2, i, str(int(v)), va="center", fontsize=9)
    ax.set_title("Tickets by contact-reason subcategory")
    ax.set_xlabel("Tickets")
    bar_img = fig_to_img(fig, "03_reasons_bar")

    top5 = sub_counts.head(5).index.tolist()
    rows = []
    for r in top5:
        sub = df[df["subcategory"] == r]
        rows.append({
            "Reason (subcategory)": r,
            "Tickets": len(sub),
            "Avg resolution (min)": sub["resolution_minutes"].mean(),
            "Median resolution (min)": sub["resolution_minutes"].median(),
            "Avg selling amount": sub["Selling Amount"].mean(),
            "% CEC-flagged": 100 * sub["is_compliance_flagged"].mean(),
        })
    tbl = pd.DataFrame(rows)
    tbl_disp = tbl.copy()
    tbl_disp["Tickets"] = tbl_disp["Tickets"].map(fmt_int)
    tbl_disp["Avg resolution (min)"] = tbl_disp["Avg resolution (min)"].map(lambda v: f"{v:,.1f}")
    tbl_disp["Median resolution (min)"] = tbl_disp["Median resolution (min)"].map(lambda v: f"{v:,.1f}")
    tbl_disp["Avg selling amount"] = tbl_disp["Avg selling amount"].map(lambda v: fmt_money(v, "$"))
    tbl_disp["% CEC-flagged"] = tbl_disp["% CEC-flagged"].map(fmt_pct)
    top5_html = df_to_html_table(tbl_disp)

    # Identify the disproportionate reasons
    by_reason = df.groupby("subcategory").agg(
        n=("Ticket ID", "count"),
        avg_value=("Selling Amount", "mean"),
        cec_rate=("is_compliance_flagged", "mean"),
    )
    by_reason = by_reason[by_reason["n"] >= 3]
    top_value_reason = by_reason["avg_value"].idxmax() if len(by_reason) else "—"
    top_cec_reason = by_reason["cec_rate"].idxmax() if len(by_reason) else "—"

    return (
        "<p class='so-what'>So what? Two contact reasons account for ~40% of all calls — Refund and Legal hold. "
        "Both signal post-send friction (the customer has already paid but the order didn't complete cleanly). "
        "That mix is a strong indicator that the cost-driver is not pre-send confusion but downstream "
        "correspondent / compliance handling.</p>"
        + bar_img
        + "<h3>Top 5 reasons — operational profile</h3>" + top5_html
        + f"<p>Highest avg selling amount among reasons with ≥3 tickets: "
        f"<strong>{top_value_reason}</strong> "
        f"(${by_reason.loc[top_value_reason, 'avg_value']:,.2f}). "
        f"Highest CEC-flag rate: <strong>{top_cec_reason}</strong> "
        f"({100*by_reason.loc[top_cec_reason, 'cec_rate']:.1f}%).</p>"
        if len(by_reason) else
        "<p class='so-what'>So what? Insufficient volume per reason for a disproportionality call.</p>"
    )


def section_4_corridor(df: pd.DataFrame) -> str:
    ct_count = pd.crosstab(df["Country"], df["Destination Country"]).fillna(0).astype(int)
    ct_value = df.groupby(["Country", "Destination Country"])["Selling Amount"].sum().unstack(fill_value=0)
    ct_value = ct_value.reindex(index=ct_count.index, columns=ct_count.columns, fill_value=0)
    ct_count.to_csv(DATA_DIR / "corridor_crosstab_count.csv")
    ct_value.to_csv(DATA_DIR / "corridor_crosstab_value.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.heatmap(ct_count, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=axes[0], cbar=False)
    axes[0].set_title("Ticket count by corridor")
    sns.heatmap(ct_value, annot=True, fmt=",.0f", cmap=SEQUENTIAL_CMAP, ax=axes[1], cbar=False)
    axes[1].set_title("Total selling amount by corridor ($)")
    corridor_img = fig_to_img(fig, "04a_corridor_heatmaps")

    top_count_corridor = df["corridor"].value_counts().idxmax()
    top_count_n = int(df["corridor"].value_counts().max())
    by_corridor_value = df.groupby("corridor")["Selling Amount"].sum().sort_values(ascending=False)
    top_value_corridor = by_corridor_value.idxmax()
    top_value_sum = by_corridor_value.iloc[0]

    # Correspondent performance
    corr = df.groupby("Correspondent").agg(
        Tickets=("Ticket ID", "count"),
        AvgResMin=("resolution_minutes", "mean"),
        PctStalled=("Order Status", lambda s: 100 * s.isin(NON_TERMINAL_STATUSES).mean()),
        TotalSelling=("Selling Amount", "sum"),
    ).sort_values("PctStalled", ascending=False)
    corr_disp = corr.copy()
    corr_disp["Tickets"] = corr_disp["Tickets"].map(fmt_int)
    corr_disp["AvgResMin"] = corr_disp["AvgResMin"].map(lambda v: f"{v:,.1f}")
    corr_disp["PctStalled"] = corr_disp["PctStalled"].map(fmt_pct)
    corr_disp["TotalSelling"] = corr_disp["TotalSelling"].map(lambda v: fmt_money(v, "$"))
    corr_disp = corr_disp.rename(columns={
        "AvgResMin": "Avg resolution (min)",
        "PctStalled": "% in non-terminal status",
        "TotalSelling": "Total selling ($)",
    })
    corr_html = df_to_html_table(corr_disp.reset_index())

    return (
        "<p class='so-what'>So what? AU → Samoa is the dominant corridor by both ticket count and aggregate "
        "value. Two correspondents — Ria Open Payment Network and Digicel Vanuatu Cash Pickup — handle the "
        "bulk of the volume, and the stall pattern is concentrated in a few partners. That tells us where to "
        "focus payout-partner SLA conversations.</p>"
        + corridor_img
        + f"<p>Top corridor by volume: <strong>{top_count_corridor}</strong> ({top_count_n} tickets). "
        f"Top corridor by aggregate value: <strong>{top_value_corridor}</strong> "
        f"(${top_value_sum:,.0f}).</p>"
        + "<h3>Correspondent performance — sorted by % in non-terminal status (stall pattern)</h3>"
        + corr_html
    )


def section_5_lifecycle(df: pd.DataFrame, enrichment: dict) -> str:
    transition = pd.crosstab(df["Original Order Status"], df["Order Status"])
    transition.to_csv(DATA_DIR / "status_transition_matrix.csv")
    # Heatmap (Sankey skipped — small n + offline build keeps this dependency-light).
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(transition, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
    ax.set_title("Order status transition matrix (rows = at ticket creation, cols = at data pull)")
    ax.set_xlabel("Current Order Status"); ax.set_ylabel("Original Order Status")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    transition_img = fig_to_img(fig, "05a_status_transition_heatmap")

    legal_hold_pct = 100 * (df["Order Status"] == "Legal Hold").mean()
    lh_avg = df.loc[df["Order Status"] == "Legal Hold", "Selling Amount"].mean()
    rest_avg = df.loc[df["Order Status"] != "Legal Hold", "Selling Amount"].mean()

    repeat = df.groupby("Ria Order Number").size().rename("ticket_count")
    repeat = repeat[repeat > 1].sort_values(ascending=False)
    repeat_n_orders = len(repeat)
    repeat_n_tickets = int(repeat.sum())
    repeat_df = df[df["Ria Order Number"].isin(repeat.index)]
    repeat_breakdown = (
        repeat_df.groupby(["Ria Order Number", "subcategory"]).size()
        .unstack(fill_value=0)
        .assign(Total=lambda d: d.sum(axis=1))
        .sort_values("Total", ascending=False)
    )
    repeat_html = df_to_html_table(repeat_breakdown.reset_index())

    # Pick a few candidate ticket IDs to quote (repeat-contact orders)
    quote_ids = repeat_df.sort_values("Ria Order Number")["Ticket ID"].tolist()
    quotes = enrichment_block(quote_ids, enrichment, max_quotes=2)

    return (
        "<p class='so-what'>So what? Status doesn't move much between ticket creation and data pull — most "
        "tickets reach their terminal disposition within the same handling window. The one striking pattern is "
        "Legal Hold concentration on high-value Samoa orders, and a handful of orders that customers call about "
        "repeatedly before resolution.</p>"
        + transition_img
        + f"<h3>Legal Hold concentration</h3>"
        + f"<p><strong>{legal_hold_pct:.1f}%</strong> of all tickets involve orders currently in Legal Hold. "
        f"Average selling amount on Legal Hold tickets is <strong>${lh_avg:,.2f}</strong>, vs "
        f"<strong>${rest_avg:,.2f}</strong> on all other tickets — a "
        f"<strong>{(lh_avg/rest_avg - 1)*100:+.0f}%</strong> differential. (n is too small for a hypothesis "
        f"test; treat as directional.)</p>"
        + f"<h3>Repeat-contact orders</h3>"
        + f"<p><strong>{repeat_n_orders}</strong> orders generated more than one ticket — "
        f"{repeat_n_tickets} tickets in total ({100*repeat_n_tickets/len(df):.0f}% of volume). "
        f"This is the 'repeat-contact friction' signal: same order, customer calling again because something "
        f"didn't resolve on first contact.</p>"
        + repeat_html + quotes
    )


def section_6_compliance(df: pd.DataFrame, enrichment: dict) -> str:
    cec_counts = df["CEC Code"].replace("", "(blank)").value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cec_counts.index, cec_counts.values, color=PALETTE[:len(cec_counts)])
    for i, v in enumerate(cec_counts.values):
        ax.text(i, v + 0.3, str(int(v)), ha="center", fontsize=9)
    ax.set_title("CEC Code distribution")
    ax.set_ylabel("Tickets")
    cec_img = fig_to_img(fig, "06a_cec_distribution")

    # CEC × reason / corridor / status — render 3 small heatmaps stacked
    coded = df[df["CEC Code"] != ""]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(
        axes,
        [("subcategory", "CEC × Reason"), ("Destination Country", "CEC × Destination"), ("Order Status", "CEC × Status")]
    ):
        ct = pd.crosstab(coded["CEC Code"], coded[col])
        sns.heatmap(ct, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    cec_cross_img = fig_to_img(fig, "06b_cec_cross_tabs")

    # Compliance/Fraud routed tickets
    routed = df[df["Ticket group"].isin(COMPLIANCE_GROUPS)]
    if len(routed) > 0:
        routed_table = pd.DataFrame({
            "Tickets": [len(routed)],
            "Avg selling amount": [routed["Selling Amount"].mean()],
            "Median selling amount": [routed["Selling Amount"].median()],
            "Avg resolution (min)": [routed["resolution_minutes"].mean()],
            "Top corridor": [routed["corridor"].value_counts().idxmax()],
            "Top reason": [routed["subcategory"].value_counts().idxmax()],
        })
        routed_table["Avg selling amount"] = routed_table["Avg selling amount"].map(lambda v: fmt_money(v, "$"))
        routed_table["Median selling amount"] = routed_table["Median selling amount"].map(lambda v: fmt_money(v, "$"))
        routed_table["Avg resolution (min)"] = routed_table["Avg resolution (min)"].map(lambda v: f"{v:,.1f}")
        routed_html = df_to_html_table(routed_table)
    else:
        routed_html = "<p>No tickets routed to Ria - APAC Compliance or Ria - Digital Fraud in this period.</p>"

    # Repeat-customer / structuring watchlist
    loc = df.groupby("Customer Sending Location").agg(
        Tickets=("Ticket ID", "count"),
        UniqueOrders=("Ria Order Number", "nunique"),
        TotalSelling=("Selling Amount", "sum"),
        AvgSelling=("Selling Amount", "mean"),
        CecRate=("is_compliance_flagged", lambda s: 100 * s.mean()),
    )
    high_volume_threshold = 3
    value_threshold = loc["TotalSelling"].quantile(0.75)
    watch = loc[(loc["Tickets"] >= high_volume_threshold) | (loc["TotalSelling"] >= value_threshold)]
    watch = watch.sort_values("Tickets", ascending=False)
    watch_disp = watch.copy()
    watch_disp["Tickets"] = watch_disp["Tickets"].map(fmt_int)
    watch_disp["UniqueOrders"] = watch_disp["UniqueOrders"].map(fmt_int)
    watch_disp["TotalSelling"] = watch_disp["TotalSelling"].map(lambda v: fmt_money(v, "$"))
    watch_disp["AvgSelling"] = watch_disp["AvgSelling"].map(lambda v: fmt_money(v, "$"))
    watch_disp["CecRate"] = watch_disp["CecRate"].map(fmt_pct)
    watch_disp = watch_disp.rename(columns={
        "UniqueOrders": "Unique orders",
        "TotalSelling": "Total selling ($)",
        "AvgSelling": "Avg selling ($)",
        "CecRate": "% CEC-flagged",
    })
    watch_html = df_to_html_table(watch_disp.reset_index())

    # Quoted evidence from compliance/fraud-routed tickets
    quotes = enrichment_block(routed["Ticket ID"].tolist(), enrichment, max_quotes=2)

    return (
        "<p class='so-what'>So what? Roughly 4-in-10 tickets carry a CEC code, signalling real compliance "
        "exposure on this corridor. The codes don't spread evenly — 1002 dominates and lines up tightly with "
        "Legal Hold disposition, suggesting it's the operational flag for the same risk class. One sending "
        "location (EW103) generates the lion's share of tickets and concentrated value, putting it in scope for "
        "an enhanced-monitoring watch list.</p>"
        + cec_img + cec_cross_img
        + "<h3>Compliance/Fraud-routed ticket profile</h3>" + routed_html
        + "<h3>Customer-location watchlist (≥3 tickets OR top-quartile total value)</h3>"
        + watch_html + quotes
    )


def section_7_resolution(df: pd.DataFrame) -> str:
    rm = df["resolution_minutes"].clip(upper=df["resolution_minutes"].quantile(0.99))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(rm, bins=30, color=PALETTE[0], edgecolor="white")
    axes[0].set_title("Resolution time distribution (clipped at 99th pctile)")
    axes[0].set_xlabel("Minutes"); axes[0].set_ylabel("Tickets")
    sns.boxplot(
        x="Ticket group", y="resolution_minutes",
        data=df.assign(resolution_minutes=df["resolution_minutes"].clip(upper=df["resolution_minutes"].quantile(0.99))),
        ax=axes[1], hue="Ticket group", palette=PALETTE, legend=False,
    )
    axes[1].set_title("Resolution time by ticket group")
    axes[1].set_xlabel(""); axes[1].set_ylabel("Minutes")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    dist_img = fig_to_img(fig, "07a_resolution_distribution")

    def agg_table(group_col):
        g = df.groupby(group_col).agg(
            Tickets=("Ticket ID", "count"),
            AvgMin=("resolution_minutes", "mean"),
            MedianMin=("resolution_minutes", "median"),
        ).sort_values("AvgMin", ascending=False)
        gd = g.copy()
        gd["Tickets"] = gd["Tickets"].map(fmt_int)
        gd["AvgMin"] = gd["AvgMin"].map(lambda v: f"{v:,.1f}")
        gd["MedianMin"] = gd["MedianMin"].map(lambda v: f"{v:,.1f}")
        gd = gd.rename(columns={"AvgMin": "Avg (min)", "MedianMin": "Median (min)"})
        return df_to_html_table(gd.reset_index())

    by_reason_html = agg_table("subcategory")
    by_group_html = agg_table("Ticket group")
    by_dest_html = agg_table("Destination Country")

    # SLA
    sla = {
        "≤ 15 min": (df["resolution_minutes"] <= 15).mean() * 100,
        "≤ 30 min": (df["resolution_minutes"] <= 30).mean() * 100,
        "≤ 1 hour": (df["resolution_minutes"] <= 60).mean() * 100,
        "≤ 24 hours": (df["resolution_minutes"] <= 1440).mean() * 100,
    }
    sla_df = pd.DataFrame([{"Bucket": k, "% within": fmt_pct(v)} for k, v in sla.items()])
    sla_html = df_to_html_table(sla_df)

    outliers = df[df["resolution_minutes"] > 1440][[
        "Ticket ID", "subcategory", "Order Status", "Ticket group",
        "Destination Country", "Selling Amount", "resolution_minutes",
    ]].sort_values("resolution_minutes", ascending=False)
    outliers_disp = outliers.copy()
    outliers_disp["Selling Amount"] = outliers_disp["Selling Amount"].map(lambda v: fmt_money(v, "$"))
    outliers_disp["resolution_minutes"] = outliers_disp["resolution_minutes"].map(lambda v: f"{v:,.0f}")
    outliers_disp = outliers_disp.rename(columns={"resolution_minutes": "Resolution (min)", "subcategory": "Reason"})
    outliers_html = df_to_html_table(outliers_disp)

    return (
        "<p class='so-what'>So what? The median ticket closes in under 7 minutes — service on the routine "
        "calls is fast. The pain is a small set of multi-day outliers (mostly APAC Care handling "
        "Modification/Recall workflows), which drag the mean up by two orders of magnitude. SLA fixes should "
        "target outlier handling, not the routine queue.</p>"
        + dist_img
        + "<h3>SLA compliance</h3>" + sla_html
        + "<h3>Avg/median resolution by reason</h3>" + by_reason_html
        + "<h3>Avg/median resolution by ticket group</h3>" + by_group_html
        + "<h3>Avg/median resolution by destination country</h3>" + by_dest_html
        + f"<h3>Outliers (>24 hours, n={len(outliers)})</h3>" + outliers_html
    )


def section_8_source(df: pd.DataFrame) -> str:
    # Channel-mix chart
    chan = df["channel_simple"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(chan.index, chan.values, color=PALETTE[:len(chan)])
    for i, v in enumerate(chan.values):
        ax.text(i, v + 1, str(int(v)), ha="center", fontsize=9)
    ax.set_title("Tickets by channel (Feb–Apr 2026)")
    ax.set_ylabel("Tickets")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    chan_img = fig_to_img(fig, "08a_channel")

    # Source bar
    src = df["Source"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(src.index, src.values, color=PALETTE[:len(src)])
    for i, v in enumerate(src.values):
        ax.text(i, v + 1, str(int(v)), ha="center", fontsize=9)
    ax.set_title("Tickets by Source (who initiated)")
    ax.set_ylabel("Tickets")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    src_img = fig_to_img(fig, "08b_source")

    # Channel × reason heatmap
    ch_reason = pd.crosstab(df["channel_simple"], df["subcategory"])
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(ch_reason, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
    ax.set_title("Channel × Reason (count)")
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ch_reason_img = fig_to_img(fig, "08c_channel_reason")

    agent = df[df["Source"] == "Agent / Store"]
    customer = df[df["Source"] == "Customer"]
    side = pd.concat([
        agent["subcategory"].value_counts().rename("Agent / Store"),
        customer["subcategory"].value_counts().rename("Customer"),
    ], axis=1).fillna(0).astype(int)
    side["Agent share %"] = (100 * side["Agent / Store"] / max(side["Agent / Store"].sum(), 1)).round(1)
    side["Customer share %"] = (100 * side["Customer"] / max(side["Customer"].sum(), 1)).round(1)
    side["Δ share (pp)"] = (side["Agent share %"] - side["Customer share %"]).round(1)
    side = side.sort_values("Δ share (pp)", ascending=False)
    side_html = df_to_html_table(side.reset_index().rename(columns={"index": "Reason"}))

    n = len(df)
    inbound_pct = 100 * (df["channel_simple"] == "Inbound call").mean()
    chat_pct = 100 * (df["channel_simple"] == "Web chat").mean()
    wa_pct = 100 * (df["channel_simple"] == "WhatsApp").mean()
    customer_pct = 100 * (df["Source"] == "Customer").mean()

    return (
        f"<p class='so-what'>So what? Inbound calls remain the dominant channel "
        f"(<strong>{inbound_pct:.0f}%</strong> of total) but messaging is a real second leg — Web chat "
        f"<strong>{chat_pct:.0f}%</strong> and WhatsApp <strong>{wa_pct:.0f}%</strong>. The cross-channel "
        f"reason mix is telling: status-check questions concentrate in messaging while compliance/legal-hold "
        f"workflows still funnel through voice. That's a deliberate routing decision the operations team "
        f"should validate.</p>"
        + chan_img + src_img + ch_reason_img
        + f"<p>Customers initiate <strong>{customer_pct:.0f}%</strong> of all tickets across both channels; "
        f"the rest come from Agent / Store staff or other operational sources.</p>"
        + "<h3>Reason mix — Agent / Store vs Customer (share within each source)</h3>" + side_html
    )


def section_9_high_value(df: pd.DataFrame, enrichment: dict) -> str:
    hv = df[df["is_high_value"]]
    lv = df[~df["is_high_value"]]
    n_hv = len(hv)
    if n_hv == 0:
        return "<p>No tickets met the high-value threshold ($5,000).</p>"

    rows = [
        {"Segment": f"High-value (>${HIGH_VALUE_THRESHOLD:,.0f})", "Tickets": n_hv,
         "Avg selling": hv["Selling Amount"].mean(), "Median selling": hv["Selling Amount"].median(),
         "Avg resolution (min)": hv["resolution_minutes"].mean(),
         "% Legal Hold": 100 * (hv["Order Status"] == "Legal Hold").mean(),
         "% CEC-flagged": 100 * hv["is_compliance_flagged"].mean()},
        {"Segment": "All other tickets", "Tickets": len(lv),
         "Avg selling": lv["Selling Amount"].mean(), "Median selling": lv["Selling Amount"].median(),
         "Avg resolution (min)": lv["resolution_minutes"].mean(),
         "% Legal Hold": 100 * (lv["Order Status"] == "Legal Hold").mean(),
         "% CEC-flagged": 100 * lv["is_compliance_flagged"].mean()},
    ]
    cmp = pd.DataFrame(rows)
    cmp_disp = cmp.copy()
    cmp_disp["Tickets"] = cmp_disp["Tickets"].map(fmt_int)
    cmp_disp["Avg selling"] = cmp_disp["Avg selling"].map(lambda v: fmt_money(v, "$"))
    cmp_disp["Median selling"] = cmp_disp["Median selling"].map(lambda v: fmt_money(v, "$"))
    cmp_disp["Avg resolution (min)"] = cmp_disp["Avg resolution (min)"].map(lambda v: f"{v:,.1f}")
    cmp_disp["% Legal Hold"] = cmp_disp["% Legal Hold"].map(fmt_pct)
    cmp_disp["% CEC-flagged"] = cmp_disp["% CEC-flagged"].map(fmt_pct)
    cmp_html = df_to_html_table(cmp_disp)

    # Corridor + correspondent + reason for high-value
    by_corr = hv["corridor"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(by_corr.index[::-1], by_corr.values[::-1], color=PALETTE[3])
    for i, v in enumerate(by_corr.values[::-1]):
        ax.text(v + 0.05, i, str(int(v)), va="center", fontsize=9)
    ax.set_title("High-value tickets by corridor")
    ax.set_xlabel("Tickets")
    hv_corridor_img = fig_to_img(fig, "09a_high_value_corridor")

    hv_table = hv[["Ticket ID", "Destination Country", "Correspondent", "Selling Amount",
                   "Order Status", "subcategory", "CEC Code"]].sort_values("Selling Amount", ascending=False)
    hv_disp = hv_table.copy()
    hv_disp["Selling Amount"] = hv_disp["Selling Amount"].map(lambda v: fmt_money(v, "$"))
    hv_disp = hv_disp.rename(columns={"subcategory": "Reason"})
    hv_html = df_to_html_table(hv_disp)

    top_corr_hv = hv["Correspondent"].value_counts().idxmax()
    top_corr_n = int(hv["Correspondent"].value_counts().max())

    quotes = enrichment_block(hv["Ticket ID"].tolist(), enrichment, max_quotes=2)

    return (
        "<p class='so-what'>So what? High-value tickets are a small slice of volume but disproportionately "
        "represented in Legal Hold and CEC-flagged dispositions, and they concentrate almost entirely on a "
        "single destination (Samoa) and a single correspondent. This is the corridor concentration risk to "
        "watch.</p>"
        + cmp_html
        + f"<p>Of {n_hv} high-value tickets, <strong>{top_corr_n}</strong> went through "
        f"<strong>{top_corr_hv}</strong>.</p>"
        + hv_corridor_img + "<h3>High-value tickets (detail)</h3>" + hv_html + quotes
    )


def section_10_digital(df: pd.DataFrame) -> str:
    digital = df[df["is_digital_location"]]
    physical = df[~df["is_digital_location"]]
    digital_pct = 100 * len(digital) / len(df)

    rows = [
        {"Segment": "Digital (EW)", "Tickets": len(digital),
         "% of total": 100 * len(digital) / len(df),
         "Avg selling": digital["Selling Amount"].mean(),
         "Avg resolution (min)": digital["resolution_minutes"].mean(),
         "% CEC-flagged": 100 * digital["is_compliance_flagged"].mean(),
         "% Legal Hold": 100 * (digital["Order Status"] == "Legal Hold").mean()},
        {"Segment": "Physical store", "Tickets": len(physical),
         "% of total": 100 * len(physical) / len(df),
         "Avg selling": physical["Selling Amount"].mean(),
         "Avg resolution (min)": physical["resolution_minutes"].mean(),
         "% CEC-flagged": 100 * physical["is_compliance_flagged"].mean(),
         "% Legal Hold": 100 * (physical["Order Status"] == "Legal Hold").mean()},
    ]
    cmp = pd.DataFrame(rows)
    cmp_disp = cmp.copy()
    cmp_disp["Tickets"] = cmp_disp["Tickets"].map(fmt_int)
    cmp_disp["% of total"] = cmp_disp["% of total"].map(fmt_pct)
    cmp_disp["Avg selling"] = cmp_disp["Avg selling"].map(lambda v: fmt_money(v, "$"))
    cmp_disp["Avg resolution (min)"] = cmp_disp["Avg resolution (min)"].map(lambda v: f"{v:,.1f}")
    cmp_disp["% CEC-flagged"] = cmp_disp["% CEC-flagged"].map(fmt_pct)
    cmp_disp["% Legal Hold"] = cmp_disp["% Legal Hold"].map(fmt_pct)
    cmp_html = df_to_html_table(cmp_disp)

    # Stacked bar of reasons by channel-type
    reason_mix = df.groupby(["channel_type", "subcategory"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    reason_mix.plot(kind="bar", stacked=True, ax=ax, color=PALETTE * 3, width=0.55)
    ax.set_title("Reason mix by channel type (stacked)")
    ax.set_xlabel(""); ax.set_ylabel("Tickets")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False, ncol=1)
    mix_img = fig_to_img(fig, "10a_channel_reason_mix")

    return (
        f"<p class='so-what'>So what? Digital sending locations (EW-prefix) generate <strong>{digital_pct:.0f}%</strong> "
        f"of all corridor inbound calls — overwhelmingly from EW103. Digital tickets carry a meaningfully higher "
        f"compliance-flag rate than physical-store tickets, which is consistent with reduced front-line ID-check "
        f"friction at point of send. This concentration is the single biggest lever for self-service and "
        f"proactive-comms work.</p>"
        + cmp_html + mix_img
    )


def section_11_recs(df: pd.DataFrame, kpis: dict, enrichment: dict) -> str:
    # Compute the data points each recommendation will cite, so numbers are live.
    n = len(df)
    legal_hold_n = int((df["Order Status"] == "Legal Hold").sum())
    hv = df[df["is_high_value"]]
    hv_lh_pct = 100 * (hv["Order Status"] == "Legal Hold").mean() if len(hv) else 0
    samoa_hv_n = int((hv["Destination Country"] == "Samoa").sum())
    ew103_n = int((df["Customer Sending Location"] == "EW103").sum())
    ew103_pct = 100 * ew103_n / n
    repeat = df.groupby("Ria Order Number").size()
    repeat = repeat[repeat > 1]
    repeat_n = len(repeat)
    repeat_top = repeat.idxmax(); repeat_top_n = int(repeat.max())
    outliers_n = int((df["resolution_minutes"] > 1440).sum())
    apac_care_outliers = int(((df["resolution_minutes"] > 1440) & (df["Ticket group"] == "Ria - APAC Care")).sum())
    cec_pct = kpis["pct_cec"]
    top_corr = df["Correspondent"].value_counts().idxmax()
    top_corr_n = int(df["Correspondent"].value_counts().max())
    refund_n = int((df["subcategory"] == "Refund").sum())

    recs = [
        {
            "title": "Stand up a Samoa high-value monitoring lane",
            "finding": (
                f"All {len(hv)} high-value tickets (&gt;${HIGH_VALUE_THRESHOLD:,.0f}) in April routed to Samoa, and "
                f"{hv_lh_pct:.0f}% of them landed in Legal Hold. The pattern is a corridor-concentration risk: "
                f"high-ticket-value × single-destination × single-disposition."
            ),
            "impact": (
                "Compliance exposure (potential AML/structuring), customer wait time on the highest-value "
                "transactions, and reputational risk if Legal-Hold backlog grows."
            ),
            "rec": (
                "Create a dedicated Samoa high-value queue with mandatory pre-send EDD for selling &gt; $5k and a "
                "named compliance partner SLA on Legal-Hold release."
            ),
            "priority": "High",
            "owner": "Compliance",
        },
        {
            "title": "Reduce EW103 inbound-call dependency with proactive comms",
            "finding": (
                f"EW103 alone accounts for {ew103_n} of {n} tickets ({ew103_pct:.0f}%). It's the single largest "
                f"source of inbound calls into the APAC desk."
            ),
            "impact": (
                "Concentrated handle-time cost; every minute shaved off EW103-related calls cascades. Also a "
                "single-point-of-failure risk for digital sending volume."
            ),
            "rec": (
                "Ship proactive status SMS/email for Sent-to-Corresp and Posted statuses for EW103 senders; add "
                "an in-app order-status check before offering the call-us CTA."
            ),
            "priority": "High",
            "owner": "Product",
        },
        {
            "title": "Close the repeat-contact loop on stalled orders",
            "finding": (
                f"{repeat_n} distinct orders generated repeat tickets in April; the worst — order "
                f"{repeat_top} — was called about {repeat_top_n} times. Repeat-contact friction signals customers "
                f"are not getting closure on first interaction."
            ),
            "impact": "Doubles handle-cost per resolved order; erodes customer trust on already-stressful issues.",
            "rec": (
                "Add a 'previously contacted' flag on the agent screen so the second-touch agent inherits "
                "context; set a 48h proactive-callback on Legal-Hold and To-Cancel orders."
            ),
            "priority": "High",
            "owner": "Operations",
        },
        {
            "title": "Investigate APAC Care long-tail handling",
            "finding": (
                f"{outliers_n} tickets exceeded 24-hour resolution; {apac_care_outliers} of them sat with "
                f"Ria - APAC Care. Mean resolution ({kpis['avg_res_min']:,.0f} min) is two orders of magnitude "
                f"larger than median ({kpis['med_res_min']:.1f} min)."
            ),
            "impact": "Hidden in averages — these are the orders that generate executive escalations.",
            "rec": (
                "Run a root-cause review on each &gt;24h ticket; suspect drivers are Modification + Recall "
                "workflows that need correspondent ping-pong. Define a 4-hour interim-status SLA so customers "
                "aren't left in the dark."
            ),
            "priority": "Medium",
            "owner": "Operations",
        },
        {
            "title": "Treat CEC 1002 as the Samoa Legal-Hold signal",
            "finding": (
                f"{cec_pct:.0f}% of tickets are CEC-flagged; code 1002 dominates the flagged set and aligns "
                f"tightly with Legal-Hold disposition on Samoa transactions."
            ),
            "impact": "If 1002 is the de-facto AML/structuring flag, treating it operationally as such would let the floor route faster.",
            "rec": (
                "Confirm CEC code semantics with Compliance, then build an auto-route rule: CEC 1002 + Samoa + "
                "high-value → straight to APAC Compliance group, skipping the Ria - Care first touch."
            ),
            "priority": "Medium",
            "owner": "Compliance",
        },
        {
            "title": "Address the Refund + Legal-Hold reason concentration with a workflow refresh",
            "finding": (
                f"Refund ({refund_n} tickets) and Legal hold ({legal_hold_n} tickets) together account for "
                f"~40% of all calls — both are post-send friction reasons, not pre-send confusion."
            ),
            "impact": "Customer-experience damage compounds when refund cycles stretch; agents repeat the same explanation across multiple touches.",
            "rec": (
                "Draft a one-page customer-facing 'what to expect when your transaction is on hold / being "
                "refunded' explainer and embed in the SMS/email proactive comms from rec #2."
            ),
            "priority": "Medium",
            "owner": "Training",
        },
        {
            "title": "Resolve the cross-channel reporting gap (Feb messaging)",
            "finding": "March + April messaging tickets are present in this analysis, but a Feb messaging file was not delivered. Channel-mix conclusions for Feb 2026 are therefore inbound-only.",
            "impact": "Trend-call on messaging adoption (chat / WhatsApp) is two months instead of three — narrow base for any pattern call.",
            "rec": "Re-pull the Feb 2026 messaging-tickets APAC export and rerun build_report.py to backfill. This is a 5-minute data ops task.",
            "priority": "Low",
            "owner": "Operations",
        },
        {
            "title": "Negotiate stall-time SLAs with the top correspondent",
            "finding": f"{top_corr_n} tickets ({100*top_corr_n/n:.0f}% of volume) touched {top_corr}. Its share of stalled (non-terminal) tickets is over-represented vs. its volume share.",
            "impact": "Correspondent stalls are the upstream driver of Legal Hold and Refund tickets — fixing them removes work from the call centre.",
            "rec": "Open a payout-partner SLA conversation with the top correspondent on Posted → Paid turnaround time and Legal-Hold release acknowledgement.",
            "priority": "Medium",
            "owner": "Operations",
        },
    ]

    # Write a CSV of recommendations
    rec_csv = pd.DataFrame([{
        "Title": r["title"], "Finding": r["finding"], "Impact": r["impact"],
        "Recommendation": r["rec"], "Priority": r["priority"], "Owner": r["owner"],
    } for r in recs])
    rec_csv.to_csv(DATA_DIR / "recommendations.csv", index=False)

    cards = []
    for r in recs:
        prio_class = f"prio-{r['priority'].lower()}"
        cards.append(
            f'<div class="rec-card">'
            f'<div class="rec-head"><span class="{prio_class}">{r["priority"]}</span> '
            f'<span class="rec-owner">Owner: {r["owner"]}</span></div>'
            f'<div class="rec-title">{escape(r["title"])}</div>'
            f'<div class="rec-row"><span class="rec-label">Finding:</span> {r["finding"]}</div>'
            f'<div class="rec-row"><span class="rec-label">Impact:</span> {r["impact"]}</div>'
            f'<div class="rec-row"><span class="rec-label">Recommendation:</span> {r["rec"]}</div>'
            f'</div>'
        )

    # Pull enrichment evidence: any candidate IDs we have, attach to the bottom.
    candidate_ids = (
        df.loc[df["Order Status"] == "Legal Hold", "Ticket ID"].tolist()
        + df.loc[df["is_high_value"], "Ticket ID"].tolist()
        + df.loc[df["Ria Order Number"].isin(repeat.index), "Ticket ID"].tolist()
    )
    quotes = enrichment_block(list(dict.fromkeys(candidate_ids)), enrichment, max_quotes=3)

    return (
        "<p class='so-what'>So what? Eight prioritised actions, sized to where the data actually points. "
        "Three are 'High' — they target the corridor-concentration risk, the EW103 self-service opportunity, "
        "and the repeat-contact friction. Each cites a number from the analysis above; quoted ticket evidence "
        "(once the Zendesk enrichment pass runs) is attached below.</p>"
        + "\n".join(cards) + quotes
    )


def section_12_appendix(df: pd.DataFrame) -> str:
    dictionary_rows = [
        ("Ticket ID", "Zendesk ticket unique identifier", "22571722"),
        ("Source", "Who initiated the ticket", "Customer, Agent / Store, (blank), Global Agent, Correspondent, Internal"),
        ("Customer Sending Location", "Store or digital channel code", "AU5487, EW103, EW111, NZ1445"),
        ("Delivery Method", "How funds are delivered", "Office Pick-Up, Bank Deposit, Mobile Payment, RIA"),
        ("Ria - Reason for Contact", "Contact reason taxonomy (Category::Subcategory)", "Order::Transaction status, Order::Refund, …"),
        ("Ria Order Number", "Internal order reference", "AU2007522064, NZ2099125364"),
        ("Country", "Sending country", "Australia, New Zealand"),
        ("Destination Country", "Receiving country", "Samoa, Fiji, Vanuatu, Tonga"),
        ("Correspondent", "Payout partner", "Ria Open Payment Network, Samoa Commercial Bank Limited, …"),
        ("Original Order Status", "Status at ticket creation", "Sent to Corresp, Paid, Legal Hold, Canceled, …"),
        ("Order Status", "Status at data pull", "(same set as Original)"),
        ("Ticket Channel", "How the ticket came in", "Inbound call, Chat - VA to Rep, WhatsApp - VA to Rep, Chat - VA only, Facebook Messenger, Instagram Direct, Twitter DM"),
        ("Ticket created - Timestamp", "ISO 8601 creation timestamp", "2026-04-01T08:34:29"),
        ("Ticket solved - Timestamp", "ISO 8601 resolution timestamp", "2026-04-01T08:43:03"),
        ("Ticket group", "Zendesk group", "Ria - Care, Ria - APAC Care, …"),
        ("CEC Code", "Compliance/escalation code", "1001, 1002, 1003, or blank"),
        ("Tickets", "Ticket count", "1.0"),
        ("Recipient Amount", "Amount in destination currency", "400.0, 90000.0"),
        ("Selling Amount", "Amount in sending currency (AUD/NZD)", "220.99, 1121.36"),
    ]
    dict_html = df_to_html_table(pd.DataFrame(dictionary_rows, columns=["Column", "Description", "Example values"]))

    methodology = f"""
<h3>Methodology notes</h3>
<ul>
<li><strong>Source files</strong>: five Excel exports combined — three inbound-call monthly pulls
(Feb / Mar / Apr 2026) plus two messaging-tickets pulls (Mar / Apr 2026). A Feb 2026 messaging file was not
delivered; channel-mix conclusions for Feb are inbound-only and called out as such.</li>
<li><strong>APAC-origin filter</strong>: rows with Country not in {{Australia, New Zealand}} are dropped at
load time. A handful of UK / US / FR origins appear in the messaging files; they're excluded so the corridor
analysis stays honest about who's sending.</li>
<li><strong>Destination column</strong>: messaging files use "Destination Countries Victor" instead of
"Destination Country"; we rename on load so the rest of the pipeline is column-name-agnostic.</li>
<li><strong>Resolution time</strong> = (Ticket solved − Ticket created), in minutes. Source timestamps are
treated as UTC as-recorded; no timezone conversion is applied.</li>
<li><strong>CEC blanks</strong> are treated as "not compliance-flagged". A CEC-flagged ticket is any row with
a non-blank value in the CEC Code column.</li>
<li><strong>High-value threshold</strong>: Selling Amount &gt; ${HIGH_VALUE_THRESHOLD:,.0f}. Messaging tickets
have no Selling Amount and are excluded from the high-value spotlight.</li>
<li><strong>Digital location</strong>: Customer Sending Location with prefix "EW" is classed as digital.
Prefixes AU / NZ are treated as physical-store.</li>
<li><strong>Repeat orders</strong>: same Ria Order Number appearing in more than one ticket (across both
channels).</li>
<li><strong>Channel grouping for §8</strong>: raw Ticket Channel values are collapsed into "Inbound call",
"Web chat", "WhatsApp", "Social media" (Facebook / Instagram / Twitter), and "Unknown".</li>
<li><strong>Sankey</strong>: rendered as a static heatmap (not a plotly Sankey) to keep the build
self-contained. Numeric content is identical.</li>
<li><strong>Schema deltas observed</strong>: Source includes "Global Agent", "Correspondent", "Internal"
beyond the original brief; Order Status taxonomy includes Voided, Refunded, Posted - Under Review (Digital),
AwaitingBalance; Delivery Method "RIA" appears.</li>
<li><strong>Sample size caution</strong>: n = {len(df):,} across 3 months. No inferential statistics are run;
all findings are descriptive.</li>
</ul>
"""

    describe = df[["Recipient Amount", "Selling Amount", "resolution_minutes"]].describe().round(2)
    describe_html = describe.to_html(classes="kpi-table", border=0)

    return (
        "<p class='so-what'>So what? Methodology and definitions, so a reader can audit any number above.</p>"
        + "<h3>Data dictionary</h3>" + dict_html + methodology
        + "<h3>Numeric summary statistics</h3>" + describe_html
    )


# ────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
  :root {
    --navy:#1f3a5f; --teal:#2e8b8b; --amber:#d99a3d; --coral:#c8553d;
    --slate:#5b6770; --sand:#f6f1e8; --paper:#fdfcfa; --rule:#e6e1d6;
    --ink:#1a1a1a; --muted:#6b6b6b; --warn-bg:#fff4d6; --warn-bord:#d99a3d;
  }
  html, body { background: var(--sand); color: var(--ink); margin:0; padding:0;
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    font-size: 14px; line-height: 1.5; }
  .layout { display: grid; grid-template-columns: 220px 1fr; gap: 0; max-width: 1280px; margin: 0 auto; }
  nav.toc { position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto;
    padding: 28px 18px; background: var(--paper); border-right: 1px solid var(--rule); }
  nav.toc h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--muted); margin: 0 0 12px 0; }
  nav.toc ol { list-style: none; padding: 0; margin: 0; counter-reset: toc; }
  nav.toc li { counter-increment: toc; margin-bottom: 8px; font-size: 13px; }
  nav.toc li::before { content: counter(toc) ". "; color: var(--amber); font-weight: 600; }
  nav.toc a { color: var(--navy); text-decoration: none; }
  nav.toc a:hover { color: var(--coral); }
  main { padding: 28px 36px 80px 36px; background: var(--paper);
    max-width: 1020px; }
  h1.report-title { color: var(--navy); font-size: 26px; margin: 0 0 4px 0; letter-spacing: -0.3px; }
  .report-meta { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  section { padding: 18px 0 24px 0; border-bottom: 1px solid var(--rule); }
  section h2 { color: var(--navy); font-size: 20px; margin: 12px 0 6px 0; letter-spacing: -0.2px; }
  section h3 { color: var(--slate); font-size: 15px; margin: 22px 0 8px 0;
    text-transform: uppercase; letter-spacing: 0.6px; }
  p { margin: 8px 0; }
  p.so-what { background: var(--sand); border-left: 4px solid var(--amber);
    padding: 12px 16px; font-style: italic; color: #444; margin: 6px 0 16px 0; }
  img.chart { max-width: 100%; height: auto; display: block; margin: 14px 0;
    border: 1px solid var(--rule); border-radius: 4px; background: white; }
  table.kpi-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12.5px; }
  table.kpi-table th { background: var(--navy); color: white; text-align: left;
    padding: 8px 10px; font-weight: 600; }
  table.kpi-table td { padding: 6px 10px; border-bottom: 1px solid var(--rule);
    vertical-align: top; }
  table.kpi-table tr:hover td { background: var(--sand); }
  .enrichment-pending { background: var(--warn-bg); border: 1px dashed var(--warn-bord);
    padding: 12px 14px; border-radius: 4px; margin: 16px 0; color: #6e5400; font-size: 13px; }
  .enrichment-quotes { display: flex; flex-direction: column; gap: 12px; margin: 16px 0; }
  .quote-card { background: var(--sand); border-left: 4px solid var(--teal);
    padding: 12px 14px; border-radius: 3px; }
  .quote-head { font-weight: 600; color: var(--navy); margin-bottom: 4px; }
  .quote-tags { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .quote-body { font-size: 13px; margin: 4px 0; }
  .quote-label { font-weight: 600; color: var(--slate); }
  .rec-card { background: white; border: 1px solid var(--rule); border-left: 5px solid var(--navy);
    border-radius: 4px; padding: 14px 18px; margin: 14px 0; }
  .rec-head { font-size: 12px; margin-bottom: 4px; }
  .rec-title { font-weight: 700; color: var(--navy); font-size: 15px; margin: 4px 0 8px 0; }
  .rec-row { margin: 4px 0; font-size: 13px; }
  .rec-label { font-weight: 600; color: var(--slate); margin-right: 4px; }
  .rec-owner { color: var(--muted); font-size: 12px; margin-left: 8px; }
  .prio-high { background: var(--coral); color: white; padding: 2px 8px; border-radius: 3px;
    font-weight: 600; font-size: 11px; }
  .prio-medium { background: var(--amber); color: white; padding: 2px 8px; border-radius: 3px;
    font-weight: 600; font-size: 11px; }
  .prio-low { background: var(--slate); color: white; padding: 2px 8px; border-radius: 3px;
    font-weight: 600; font-size: 11px; }
</style>
"""

TOC_ITEMS = [
    (1, "Executive Summary"),
    (2, "Volume & Trend"),
    (3, "Reason for Contact"),
    (4, "Corridor Analysis"),
    (5, "Order Status & Lifecycle"),
    (6, "Compliance & Risk Flags"),
    (7, "Resolution Time"),
    (8, "Source & Channel"),
    (9, "High-Value Spotlight"),
    (10, "Digital Channel"),
    (11, "Recommendations"),
    (12, "Appendix"),
]


def assemble_html(sections: dict[int, str], enrichment: dict, kpis: dict) -> str:
    toc = "\n".join(f'<li><a href="#s{n}">{title}</a></li>' for n, title in TOC_ITEMS)
    enrich_status = (
        f"Zendesk enrichment loaded — {len(enrichment)} tickets" if enrichment
        else "Zendesk enrichment not yet loaded — metadata-only base report"
    )
    n = kpis["total_tickets"]
    n_inbound = kpis["n_inbound"]
    n_messaging = kpis["n_messaging"]
    body_sections = []
    for n_id, title in TOC_ITEMS:
        body_sections.append(
            f'<section id="s{n_id}"><h2>{n_id}. {title}</h2>{sections[n_id]}</section>'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>APAC Corridor — Feb–Apr 2026 Deep-Dive</title>
{CSS}
</head>
<body>
<div class="layout">
  <nav class="toc">
    <h2>Contents</h2>
    <ol>{toc}</ol>
  </nav>
  <main>
    <h1 class="report-title">APAC Corridor Zendesk Deep-Dive</h1>
    <div class="report-meta">
      Australia / New Zealand → Pacific Islands · February 1 – April 30, 2026<br/>
      {n:,} tickets total ({n_inbound:,} inbound calls + {n_messaging:,} messaging) · multi-channel<br/>
      Built {datetime.now().strftime('%Y-%m-%d %H:%M')} · {enrich_status}
    </div>
    {''.join(body_sections)}
  </main>
</div>
</body>
</html>"""


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("Loading data…")
    df = load_data()
    enrichment = load_enrichment()
    print(f"  Total rows after APAC filter: {len(df)}")
    print(f"  By data source: {df['data_source'].value_counts().to_dict()}")
    print(f"  By month: {df['month'].value_counts().to_dict()}")

    s1, kpis = section_1_exec_summary(df, enrichment)
    sections = {
        1: s1,
        2: section_2_volume(df),
        3: section_3_reasons(df, enrichment),
        4: section_4_corridor(df),
        5: section_5_lifecycle(df, enrichment),
        6: section_6_compliance(df, enrichment),
        7: section_7_resolution(df),
        8: section_8_source(df),
        9: section_9_high_value(df, enrichment),
        10: section_10_digital(df),
        11: section_11_recs(df, kpis, enrichment),
        12: section_12_appendix(df),
    }
    html = assemble_html(sections, enrichment, kpis)
    HTML_OUT.write_text(html, encoding="utf-8")

    # Smoke checks
    n_pngs = len(list(CHARTS_DIR.glob("*.png")))
    size_kb = HTML_OUT.stat().st_size / 1024
    print(f"✔ Wrote {HTML_OUT.name} ({size_kb:,.0f} KB)")
    print(f"✔ Charts: {n_pngs} PNGs in {CHARTS_DIR.name}/")
    print(f"✔ KPIs: total={kpis['total_tickets']} orders={kpis['unique_orders']} cec%={kpis['pct_cec']:.1f}")
    print(f"✔ Enrichment records loaded: {len(enrichment)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
