"""
APAC Corridor Zendesk Deep-Dive — interactive Streamlit dashboard.

Run with:
    python3 -m streamlit run app.py

Reuses the data loader from build_report.py so the analysis stays single-source.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from build_report import (
    DATA_DIR,
    ENRICHMENT_PATH,
    HIGH_VALUE_THRESHOLD,
    NON_TERMINAL_STATUSES,
    COMPLIANCE_GROUPS,
    PALETTE,
    SEQUENTIAL_CMAP,
    DOW_ORDER,
    TRANSACTIONS_BY_DEST,
    TRANSACTIONS_PERIOD,
    load_data,
    load_enrichment,
)

st.set_page_config(
    page_title="APAC Corridor — Feb–Apr 2026",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Reuse build_report's matplotlib palette
sns.set_palette(PALETTE)
plt.rcParams.update({
    "figure.figsize": (10, 4.5), "figure.dpi": 110,
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 10,
})


# ────────────────────────────────────────────────────────────────────────────
# Data loading (cached so filters are instant)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_df() -> pd.DataFrame:
    return load_data()


@st.cache_data
def get_enrichment() -> dict:
    return load_enrichment()


df_full = get_df()
enrichment = get_enrichment()


# ────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🌏 Filters")
st.sidebar.caption("Apply to refine every section below.")

months = sorted(df_full["month"].dropna().unique().tolist())
channels = sorted(df_full["channel_simple"].dropna().unique().tolist())
destinations = sorted(df_full["Destination Country"].dropna().unique().tolist())
groups = sorted(df_full["Ticket group"].dropna().unique().tolist())
sources = sorted(df_full["Source"].dropna().unique().tolist())
reason_categories = sorted(df_full["category"].dropna().unique().tolist())
reason_subcategories = sorted(df_full["subcategory"].dropna().unique().tolist())

sel_months = st.sidebar.multiselect("Month", months, default=months)
sel_data_sources = st.sidebar.multiselect(
    "Channel category", ["inbound", "messaging"], default=["inbound", "messaging"]
)
sel_channels = st.sidebar.multiselect("Detailed channel", channels, default=channels)
sel_dests = st.sidebar.multiselect("Destination country", destinations, default=destinations)
sel_groups = st.sidebar.multiselect("Ticket group", groups, default=groups)
sel_sources = st.sidebar.multiselect("Source", sources, default=sources)
sel_reason_cats = st.sidebar.multiselect(
    "Reason category", reason_categories, default=reason_categories,
    help="The 'Category::Subcategory' before the '::' (Order, General contact, Agent requests).",
)
sel_reason_subs = st.sidebar.multiselect(
    "Reason subcategory", reason_subcategories, default=reason_subcategories,
    help="The 'Category::Subcategory' after the '::' (Refund, Legal hold, Transaction status, etc.).",
)
only_compliance = st.sidebar.checkbox("Only CEC-flagged tickets", value=False)
only_legal_hold = st.sidebar.checkbox("Only Legal Hold status", value=False)
only_high_value = st.sidebar.checkbox(f"Only high-value (> ${HIGH_VALUE_THRESHOLD:,.0f})", value=False)

df = df_full[
    df_full["month"].isin(sel_months)
    & df_full["data_source"].isin(sel_data_sources)
    & df_full["channel_simple"].isin(sel_channels)
    & df_full["Destination Country"].isin(sel_dests)
    & df_full["Ticket group"].isin(sel_groups)
    & df_full["Source"].isin(sel_sources)
    & df_full["category"].isin(sel_reason_cats)
    & df_full["subcategory"].isin(sel_reason_subs)
].copy()
if only_compliance:
    df = df[df["is_compliance_flagged"]]
if only_legal_hold:
    df = df[df["Order Status"] == "Legal Hold"]
if only_high_value:
    df = df[df["is_high_value"]]

st.sidebar.divider()
st.sidebar.metric("Tickets in current view", f"{len(df):,}", delta=f"of {len(df_full):,} total")

if len(df) == 0:
    st.warning("No tickets match the current filter combination. Loosen filters in the sidebar.")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.title("APAC Corridor Zendesk Deep-Dive")
st.caption(
    "Australia / New Zealand → Pacific Islands · February 1 – April 30, 2026 · "
    f"Multi-channel (inbound calls + chat / WhatsApp) · {len(enrichment)} tickets with Zendesk enrichment"
)


# ────────────────────────────────────────────────────────────────────────────
# Tab layout (12 sections per the original analysis)
# ────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "1. Executive Summary",
    "2. Volume & Trend",
    "3. Reasons",
    "4. Corridors",
    "5. Lifecycle",
    "6. Compliance",
    "7. Resolution",
    "8. Channel",
    "9. High-Value",
    "10. Digital",
    "11. Recommendations",
    "12. Appendix",
])


def fig_palette_axes(fig, ax):
    ax.tick_params(colors="#333")
    return fig


def with_totals(df: pd.DataFrame, rows: bool = True, cols: bool = True, label: str = "Total") -> pd.DataFrame:
    """Append a Total row and/or column to a 2-D numeric dataframe for display."""
    out = df.copy()
    if cols:
        out[label] = out.sum(axis=1, numeric_only=True)
    if rows:
        # Row of column sums (only over the numeric columns to avoid coercion issues)
        total_row = out.sum(axis=0, numeric_only=True)
        out.loc[label] = total_row
    return out


def total_caption(series_or_value, suffix: str = "tickets") -> None:
    """Render a small caption beneath a chart showing the total."""
    try:
        total = int(series_or_value.sum()) if hasattr(series_or_value, "sum") else int(series_or_value)
    except Exception:
        total = 0
    st.caption(f"**Total: {total:,} {suffix}**")


def render_enrichment_block(candidate_ids: list[int], max_quotes: int = 3):
    """Streamlit-native version of the quote-card block from build_report.py."""
    found = [(tid, enrichment[str(tid)]) for tid in candidate_ids if str(tid) in enrichment]
    if not found:
        st.info("💬 No quoted Zendesk evidence available for tickets in this filtered view.")
        return
    st.markdown("##### Quoted ticket evidence from Zendesk")
    for tid, rec in found[:max_quotes]:
        with st.expander(f"Ticket #{tid} — {rec.get('subject', '(no subject)')}", expanded=False):
            tags = rec.get("tags") or []
            if tags:
                st.caption("Tags: " + " · ".join(tags[:12]))
            first = (rec.get("first_customer_message") or "").strip()
            last = (rec.get("last_agent_reply") or "").strip()
            if first:
                st.markdown(f"**First customer message:** _{first[:500]}_")
            if last and last != first:
                st.markdown(f"**Last agent reply:** _{last[:500]}_")
            if rec.get("comments"):
                with st.expander("View full thread"):
                    for c in rec["comments"][:10]:
                        st.markdown(
                            f"**{c.get('author') or '(unknown)'}** · {c.get('timestamp') or ''}  \n"
                            f"{c.get('body', '')[:600]}"
                        )


# ────────────────────────────────────────────────────────────────────────────
# §1 Executive Summary
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        "> **So what?** One corridor (AU → Samoa), one reason category (order disposition), "
        "and one digital storefront drive the bulk of cross-channel volume — and the "
        "highest-risk tickets cluster together. Targeted intervention on those three "
        "concentrations would move the needle further than broad SLA work."
    )

    n = len(df)
    n_inbound = int((df["data_source"] == "inbound").sum())
    n_messaging = int((df["data_source"] == "messaging").sum())
    n_orders = df["Ria Order Number"].nunique()
    avg_res = df["resolution_minutes"].mean()
    med_res = df["resolution_minutes"].median()
    median_value = df["Selling Amount"].median()
    pct_cec = 100 * df["is_compliance_flagged"].mean()
    pct_legal_hold = 100 * (df["Order Status"] == "Legal Hold").mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total tickets", f"{n:,}", f"{n_inbound} call + {n_messaging} msg")
    c2.metric("Unique orders", f"{n_orders:,}")
    c3.metric("Avg resolution (min)", f"{avg_res:,.0f}" if pd.notna(avg_res) else "—",
              f"median {med_res:.1f}" if pd.notna(med_res) else "")
    c4.metric("Median selling", f"${median_value:,.2f}" if pd.notna(median_value) else "—")
    c5.metric("% CEC-flagged", f"{pct_cec:.1f}%", f"{pct_legal_hold:.1f}% in Legal Hold")

    st.markdown("---")
    st.subheader("Where the volume sits")
    all_corr = df["corridor"].value_counts()
    all_reason = df["subcategory"].value_counts()
    a, b = st.columns(2)
    with a:
        st.markdown("**Corridors**")
        st.bar_chart(all_corr, color=PALETTE[0], horizontal=True)
        st.caption(f"**Total: {int(all_corr.sum()):,} tickets across {len(all_corr)} corridors**")
    with b:
        st.markdown("**Contact reasons**")
        st.bar_chart(all_reason, color=PALETTE[1], horizontal=True)
        st.caption(f"**Total: {int(all_reason.sum()):,} tickets across {len(all_reason)} reasons**")


# ────────────────────────────────────────────────────────────────────────────
# §2 Volume & Trend
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "> **So what?** Ticket volume sits in a band of ~100–135 tickets/month, with "
        "messaging ~25–35% the size of inbound calls. Day-of-week is weekday-heavy "
        "with no real weekend tail — staffing pattern, not customer-demand pattern."
    )

    daily_split = df.pivot_table(index="date", columns="data_source", aggfunc="size", fill_value=0)
    full_index = pd.date_range(df["date"].min(), df["date"].max()).date if len(df) else []
    if len(full_index):
        daily_split = daily_split.reindex(full_index, fill_value=0)
    daily_total = daily_split.sum(axis=1)
    rolling = daily_total.rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4.2))
    cols_order = [c for c in ["inbound", "messaging"] if c in daily_split.columns]
    bottom = np.zeros(len(daily_split))
    bar_colors = {"inbound": PALETTE[0], "messaging": PALETTE[1]}
    for col in cols_order:
        ax.bar(daily_split.index, daily_split[col], bottom=bottom,
               color=bar_colors[col], label=col.capitalize(), width=0.85)
        bottom = bottom + daily_split[col].values
    ax.plot(daily_total.index, rolling.values, color=PALETTE[3], linewidth=2.5, label="7-day MA")
    ax.set_title("Daily ticket volume (stacked by channel)")
    ax.set_xlabel("Date"); ax.set_ylabel("Tickets")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    total_caption(daily_total)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Month-over-month, by channel**")
        mom = df.pivot_table(index="month", columns="data_source", aggfunc="size", fill_value=0)
        mom = mom.reindex(sorted(mom.index))
        st.bar_chart(mom, color=[PALETTE[0], PALETTE[1]][: mom.shape[1]])
        st.dataframe(with_totals(mom), use_container_width=True)
        total_caption(mom.values.sum())
    with c2:
        st.markdown("**Day of week**")
        dow = df.groupby("day_of_week").size().reindex(DOW_ORDER, fill_value=0)
        st.bar_chart(dow, color=PALETTE[1])
        total_caption(dow)
        st.markdown("**Hour of day (UTC)**")
        hour = df.groupby("hour_utc").size().reindex(range(24), fill_value=0)
        st.bar_chart(hour, color=PALETTE[4])
        total_caption(hour)


# ────────────────────────────────────────────────────────────────────────────
# §3 Reasons
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "> **So what?** Two contact reasons account for ~40% of all calls — Refund and "
        "Legal Hold. Both signal post-send friction (customer has paid but order didn't "
        "complete cleanly). Cost-driver is downstream correspondent / compliance handling, "
        "not pre-send confusion."
    )

    sub_counts = df["subcategory"].value_counts()
    st.bar_chart(sub_counts.sort_values(), horizontal=True, color=PALETTE[0])
    total_caption(sub_counts)

    st.markdown("##### Top reasons — operational profile")
    top_n = st.slider("Top N reasons", min_value=3, max_value=10, value=5, key="top_reasons_n")
    rows = []
    for r in sub_counts.head(top_n).index:
        sub = df[df["subcategory"] == r]
        rows.append({
            "Reason": r, "Tickets": len(sub),
            "Avg resolution (min)": round(sub["resolution_minutes"].mean() or 0, 1),
            "Median resolution (min)": round(sub["resolution_minutes"].median() or 0, 1),
            "Avg selling ($)": round(sub["Selling Amount"].mean() or 0, 2),
            "% CEC-flagged": round(100 * sub["is_compliance_flagged"].mean(), 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# §4 Corridors
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        "> **So what?** AU → Samoa is the dominant corridor by both ticket count and "
        "aggregate value. Two correspondents handle the bulk of volume, and the stall "
        "pattern is concentrated in a few partners."
    )

    a, b = st.columns(2)
    with a:
        st.markdown("**Tickets by corridor**")
        ct_count = pd.crosstab(df["Country"], df["Destination Country"]).fillna(0).astype(int)
        ct_count_disp = with_totals(ct_count)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(ct_count_disp, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
        ax.set_title("Ticket count"); st.pyplot(fig, use_container_width=True); plt.close(fig)
        total_caption(ct_count.values.sum())
    with b:
        st.markdown("**Total selling amount by corridor**")
        ct_value = df.groupby(["Country", "Destination Country"])["Selling Amount"].sum().unstack(fill_value=0)
        ct_value_disp = with_totals(ct_value)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(ct_value_disp, annot=True, fmt=",.0f", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
        ax.set_title("Total selling ($)"); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.caption(f"**Total: ${ct_value.values.sum():,.0f}**")

    # ──────────────────────────────────────────────────────────────────────
    # Contact rate — tickets ÷ transactions by destination country
    # Both numerator and denominator cover Feb-Apr 2026.
    # ──────────────────────────────────────────────────────────────────────
    st.markdown(f"##### Contact rate by destination country ({TRANSACTIONS_PERIOD})")
    st.caption(
        f"Contact rate = (tickets ÷ transactions) × 100. Tickets respect the current "
        f"sidebar filters; transactions come from the {TRANSACTIONS_PERIOD} operational "
        f"export. Both cover the same period."
    )
    rows = []
    for dest, vol in TRANSACTIONS_BY_DEST.items():
        dest_mask = df["Destination Country"] == dest
        n_tickets = int(dest_mask.sum())
        rate = (n_tickets / vol["transactions"] * 100) if vol["transactions"] else 0
        n_refunds = int((dest_mask & (df["subcategory"] == "Refund")).sum())
        n_lh = int((dest_mask & (df["Order Status"] == "Legal Hold")).sum())
        rows.append({
            "Destination": dest,
            "Transactions": vol["transactions"],
            "Tickets (current filter)": n_tickets,
            "Contact rate %": round(rate, 2),
            "Refund tickets": n_refunds,
            "Legal Hold tickets": n_lh,
            "Refunds (operational)": vol["refunds"],
            "Modifications (operational)": vol["modifications"],
            "Unpaid (operational)": vol["unpaids"],
        })
    rate_df = pd.DataFrame(rows).sort_values("Transactions", ascending=False)
    # Totals row
    total_txn = rate_df["Transactions"].sum()
    total_tix = rate_df["Tickets (current filter)"].sum()
    rate_df.loc[len(rate_df)] = {
        "Destination": "TOTAL",
        "Transactions": total_txn,
        "Tickets (current filter)": total_tix,
        "Contact rate %": round(total_tix / total_txn * 100, 2) if total_txn else 0,
        "Refund tickets": int(rate_df["Refund tickets"].sum()),
        "Legal Hold tickets": int(rate_df["Legal Hold tickets"].sum()),
        "Refunds (operational)": int(rate_df["Refunds (operational)"].sum()),
        "Modifications (operational)": int(rate_df["Modifications (operational)"].sum()),
        "Unpaid (operational)": int(rate_df["Unpaid (operational)"].sum()),
    }
    st.dataframe(rate_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Transactions": st.column_config.NumberColumn(format="%d"),
                     "Tickets (current filter)": st.column_config.NumberColumn(format="%d"),
                     "Contact rate %": st.column_config.NumberColumn(format="%.2f%%"),
                 })

    # Per-corridor resolution table — avg + median in minutes AND days
    st.markdown("##### Resolution time by corridor")
    corridor_res = df.groupby("corridor").agg(
        Tickets=("Ticket ID", "count"),
        AvgResMin=("resolution_minutes", "mean"),
        MedianResMin=("resolution_minutes", "median"),
        TotalSelling=("Selling Amount", "sum"),
    ).sort_values("Tickets", ascending=False)
    corridor_res["AvgResDays"] = (corridor_res["AvgResMin"] / 1440).round(2)
    corridor_res["MedianResDays"] = (corridor_res["MedianResMin"] / 1440).round(2)
    corridor_res = corridor_res.round({"AvgResMin": 1, "MedianResMin": 1, "TotalSelling": 2})
    corridor_res_disp = corridor_res[[
        "Tickets", "AvgResMin", "AvgResDays", "MedianResMin", "MedianResDays", "TotalSelling"
    ]].copy()
    corridor_res_disp.loc["Total / overall"] = [
        int(corridor_res["Tickets"].sum()),
        round(df["resolution_minutes"].mean(), 1),
        round(df["resolution_minutes"].mean() / 1440, 2),
        round(df["resolution_minutes"].median(), 1),
        round(df["resolution_minutes"].median() / 1440, 2),
        round(df["Selling Amount"].sum(), 2),
    ]
    corridor_res_disp = corridor_res_disp.rename(columns={
        "AvgResMin": "Avg resolution (min)",
        "AvgResDays": "Avg resolution (days)",
        "MedianResMin": "Median resolution (min)",
        "MedianResDays": "Median resolution (days)",
        "TotalSelling": "Total selling ($)",
    })
    st.caption("👆 Click a corridor row to drill into the underlying tickets below.")
    corridor_event = st.dataframe(
        corridor_res_disp,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="corridor_table",
    )
    if corridor_event.selection.rows:
        picked_idx = corridor_event.selection.rows[0]
        picked_corridor = corridor_res_disp.index[picked_idx]
        if picked_corridor == "Total / overall":
            drill_df = df.copy()
            drill_label = "all tickets in current filter"
        else:
            drill_df = df[df["corridor"] == picked_corridor].copy()
            drill_label = f"corridor {picked_corridor}"
        st.markdown(f"##### 🔍 {len(drill_df)} tickets — {drill_label}")
        drill_cols = [
            "Ticket ID", "month", "created", "Correspondent", "Source",
            "subcategory", "Order Status", "Ticket group",
            "Selling Amount", "resolution_minutes", "CEC Code", "channel_simple",
        ]
        drill_disp = drill_df[drill_cols].copy()
        drill_disp["created"] = drill_disp["created"].dt.strftime("%Y-%m-%d %H:%M")
        drill_disp["resolution_minutes"] = drill_disp["resolution_minutes"].round(1)
        drill_disp = drill_disp.rename(columns={
            "Ticket ID": "Ticket ID",
            "created": "Created (UTC)",
            "subcategory": "Reason",
            "Order Status": "Status",
            "Ticket group": "Group",
            "Selling Amount": "Selling ($)",
            "resolution_minutes": "Resolution (min)",
            "channel_simple": "Channel",
        })
        st.dataframe(drill_disp.sort_values("Created (UTC)", ascending=False),
                     use_container_width=True, hide_index=True)
        # Offer enrichment quotes for any of these tickets that we've scraped
        render_enrichment_block(drill_df["Ticket ID"].tolist(), max_quotes=5)

    st.markdown("##### Correspondent performance (sorted by stall rate)")
    corr = df.groupby("Correspondent").agg(
        Tickets=("Ticket ID", "count"),
        AvgResMin=("resolution_minutes", "mean"),
        MedianResMin=("resolution_minutes", "median"),
        PctStalled=("Order Status", lambda s: 100 * s.isin(NON_TERMINAL_STATUSES).mean()),
        TotalSelling=("Selling Amount", "sum"),
    ).sort_values("PctStalled", ascending=False)
    corr["AvgResDays"] = (corr["AvgResMin"] / 1440).round(2)
    corr["MedianResDays"] = (corr["MedianResMin"] / 1440).round(2)
    corr = corr.round({"AvgResMin": 1, "MedianResMin": 1, "PctStalled": 1, "TotalSelling": 0})
    corr_disp = corr[[
        "Tickets", "AvgResMin", "AvgResDays", "MedianResMin", "MedianResDays",
        "PctStalled", "TotalSelling",
    ]].copy()
    corr_disp.loc["Total / overall"] = [
        int(corr["Tickets"].sum()),
        round(df["resolution_minutes"].mean(), 1),
        round(df["resolution_minutes"].mean() / 1440, 2),
        round(df["resolution_minutes"].median(), 1),
        round(df["resolution_minutes"].median() / 1440, 2),
        round(100 * df["Order Status"].isin(NON_TERMINAL_STATUSES).mean(), 1),
        round(df["Selling Amount"].sum(), 0),
    ]
    corr_disp = corr_disp.rename(columns={
        "AvgResMin": "Avg resolution (min)",
        "AvgResDays": "Avg resolution (days)",
        "MedianResMin": "Median resolution (min)",
        "MedianResDays": "Median resolution (days)",
        "PctStalled": "% in non-terminal status",
        "TotalSelling": "Total selling ($)",
    })
    st.caption("👆 Click a correspondent row to drill into its tickets below.")
    corr_event = st.dataframe(
        corr_disp,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="correspondent_table",
    )
    if corr_event.selection.rows:
        picked_idx = corr_event.selection.rows[0]
        picked_corr = corr_disp.index[picked_idx]
        if picked_corr == "Total / overall":
            drill_df = df.copy()
            drill_label = "all tickets in current filter"
        else:
            drill_df = df[df["Correspondent"] == picked_corr].copy()
            drill_label = f"correspondent {picked_corr}"
        st.markdown(f"##### 🔍 {len(drill_df)} tickets — {drill_label}")
        drill_cols = [
            "Ticket ID", "month", "created", "corridor", "Source",
            "subcategory", "Order Status", "Ticket group",
            "Selling Amount", "resolution_minutes", "CEC Code", "channel_simple",
        ]
        drill_disp = drill_df[drill_cols].copy()
        drill_disp["created"] = drill_disp["created"].dt.strftime("%Y-%m-%d %H:%M")
        drill_disp["resolution_minutes"] = drill_disp["resolution_minutes"].round(1)
        drill_disp = drill_disp.rename(columns={
            "created": "Created (UTC)",
            "subcategory": "Reason",
            "Order Status": "Status",
            "Ticket group": "Group",
            "Selling Amount": "Selling ($)",
            "resolution_minutes": "Resolution (min)",
            "channel_simple": "Channel",
        })
        st.dataframe(drill_disp.sort_values("Created (UTC)", ascending=False),
                     use_container_width=True, hide_index=True)
        render_enrichment_block(drill_df["Ticket ID"].tolist(), max_quotes=5)


# ────────────────────────────────────────────────────────────────────────────
# §5 Lifecycle
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown(
        "> **So what?** Status doesn't move much between ticket creation and data pull — "
        "most tickets reach terminal disposition within the same handling window. The "
        "striking pattern is Legal Hold concentration on high-value Samoa orders, plus a "
        "handful of orders customers call about repeatedly."
    )

    st.markdown("##### Order status transition matrix")
    transition = pd.crosstab(df["Original Order Status"], df["Order Status"])
    transition_disp = with_totals(transition)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(transition_disp, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
    ax.set_title("Rows = at ticket creation · Columns = at data pull")
    ax.set_xlabel("Current Order Status"); ax.set_ylabel("Original Order Status")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    total_caption(transition.values.sum())

    lh = df[df["Order Status"] == "Legal Hold"]
    rest = df[df["Order Status"] != "Legal Hold"]
    a, b, c = st.columns(3)
    a.metric("% in Legal Hold", f"{100*len(lh)/max(len(df),1):.1f}%")
    b.metric("Avg selling — Legal Hold", f"${lh['Selling Amount'].mean():,.2f}" if len(lh) else "—")
    c.metric("Avg selling — all other", f"${rest['Selling Amount'].mean():,.2f}" if len(rest) else "—")

    st.markdown("##### Repeat-contact orders (same order, >1 ticket)")
    rep = df.groupby("Ria Order Number").size().rename("ticket_count")
    rep = rep[rep > 1].sort_values(ascending=False)
    st.write(f"{len(rep)} orders with multiple tickets ({int(rep.sum())} tickets total)")
    if len(rep):
        repeat_df = df[df["Ria Order Number"].isin(rep.index)]
        repeat_breakdown = (
            repeat_df.groupby(["Ria Order Number", "subcategory"]).size()
            .unstack(fill_value=0).assign(Total=lambda d: d.sum(axis=1))
            .sort_values("Total", ascending=False)
        )
        # Append totals row across order numbers
        repeat_breakdown_disp = repeat_breakdown.copy()
        repeat_breakdown_disp.loc["Total"] = repeat_breakdown.sum(axis=0)
        st.dataframe(repeat_breakdown_disp, use_container_width=True)
        render_enrichment_block(repeat_df["Ticket ID"].tolist())


# ────────────────────────────────────────────────────────────────────────────
# §6 Compliance
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(
        "> **So what?** ~1-in-3 tickets carry a CEC code. Code 1002 dominates and "
        "concentrates on Legal Hold + Samoa, suggesting it's the operational flag for "
        "the same risk class. EW103 generates the lion's share of compliance-flagged "
        "volume — enhanced-monitoring watchlist candidate."
    )

    cec_counts = df["CEC Code"].replace("", "(blank)").value_counts()
    st.bar_chart(cec_counts, color=PALETTE[0])
    total_caption(cec_counts)

    coded = df[df["CEC Code"] != ""]
    if len(coded):
        st.markdown("##### CEC code × reason / destination / status")
        a, b, c = st.columns(3)
        for col, title, container in [
            ("subcategory", "CEC × Reason", a),
            ("Destination Country", "CEC × Destination", b),
            ("Order Status", "CEC × Status", c),
        ]:
            with container:
                ct = pd.crosstab(coded["CEC Code"], coded[col])
                ct_disp = with_totals(ct)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.heatmap(ct_disp, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
                ax.set_title(title)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                total_caption(ct.values.sum())

    st.markdown("##### Compliance / Fraud-routed ticket profile")
    routed = df[df["Ticket group"].isin(COMPLIANCE_GROUPS)]
    if len(routed):
        rt = pd.DataFrame({
            "Tickets": [len(routed)],
            "Avg selling ($)": [round(routed["Selling Amount"].mean() or 0, 2)],
            "Avg resolution (min)": [round(routed["resolution_minutes"].mean() or 0, 1)],
            "Top corridor": [routed["corridor"].value_counts().idxmax()],
            "Top reason": [routed["subcategory"].value_counts().idxmax()],
        })
        st.dataframe(rt, use_container_width=True, hide_index=True)
        render_enrichment_block(routed["Ticket ID"].tolist())
    else:
        st.info("No Compliance / Fraud-routed tickets in the current filter view.")

    st.markdown("##### Customer-location watchlist (≥3 tickets OR top-quartile total value)")
    loc = df.groupby("Customer Sending Location").agg(
        Tickets=("Ticket ID", "count"),
        UniqueOrders=("Ria Order Number", "nunique"),
        TotalSelling=("Selling Amount", "sum"),
        CecRate=("is_compliance_flagged", lambda s: round(100 * s.mean(), 1)),
    )
    value_q75 = loc["TotalSelling"].quantile(0.75)
    watch = loc[(loc["Tickets"] >= 3) | (loc["TotalSelling"] >= value_q75)].sort_values("Tickets", ascending=False)
    st.dataframe(watch, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# §7 Resolution
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown(
        "> **So what?** Median ticket closes in under 7 minutes — routine service is "
        "fast. The pain is a small set of multi-day outliers (mostly APAC Care handling "
        "Modification/Recall workflows), which drag the mean up by orders of magnitude. "
        "Fixes should target outlier handling, not the routine queue."
    )

    rm = df["resolution_minutes"].dropna()
    if len(rm):
        clip = rm.clip(upper=rm.quantile(0.99))
        a, b = st.columns(2)
        with a:
            st.markdown("**Resolution time distribution (clipped at p99)**")
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.hist(clip, bins=30, color=PALETTE[0], edgecolor="white")
            ax.set_xlabel("Minutes"); ax.set_ylabel("Tickets")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with b:
            st.markdown("**Box plot by ticket group**")
            tmp = df.assign(resolution_minutes=clip)
            fig, ax = plt.subplots(figsize=(7, 3.8))
            sns.boxplot(x="Ticket group", y="resolution_minutes", data=tmp, ax=ax,
                        hue="Ticket group", palette=PALETTE, legend=False)
            ax.set_xlabel(""); ax.set_ylabel("Minutes")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("##### SLA compliance")
    sla = {
        "≤ 15 min": (df["resolution_minutes"] <= 15).mean() * 100,
        "≤ 30 min": (df["resolution_minutes"] <= 30).mean() * 100,
        "≤ 1 hour": (df["resolution_minutes"] <= 60).mean() * 100,
        "≤ 24 hours": (df["resolution_minutes"] <= 1440).mean() * 100,
    }
    c1, c2, c3, c4 = st.columns(4)
    for col, (k, v) in zip([c1, c2, c3, c4], sla.items()):
        col.metric(k, f"{v:.1f}%")

    st.markdown("##### Outliers — resolution > 24 hours")
    outliers = df[df["resolution_minutes"] > 1440][[
        "Ticket ID", "month", "subcategory", "Order Status", "Ticket group",
        "Destination Country", "Selling Amount", "resolution_minutes",
    ]].sort_values("resolution_minutes", ascending=False)
    st.write(f"{len(outliers)} tickets")
    st.dataframe(outliers, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# §8 Channel
# ────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown(
        "> **So what?** Inbound calls remain dominant but messaging is a real second leg. "
        "Cross-channel reason mix tells the routing story: status-checks lean messaging, "
        "compliance / Legal Hold still funnels through voice."
    )
    a, b = st.columns(2)
    with a:
        st.markdown("**Channel mix**")
        chan = df["channel_simple"].value_counts()
        st.bar_chart(chan, color=PALETTE[0])
        total_caption(chan)
    with b:
        st.markdown("**Source mix (who initiated)**")
        src = df["Source"].value_counts()
        st.bar_chart(src, color=PALETTE[1])
        total_caption(src)

    st.markdown("##### Channel × Reason")
    ch_reason = pd.crosstab(df["channel_simple"], df["subcategory"])
    ch_reason_disp = with_totals(ch_reason)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(ch_reason_disp, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, ax=ax, cbar=False)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    total_caption(ch_reason.values.sum())


# ────────────────────────────────────────────────────────────────────────────
# §9 High-Value
# ────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown(
        f"> **So what?** High-value tickets (> ${HIGH_VALUE_THRESHOLD:,.0f}) are a small "
        "slice of volume but disproportionately represented in Legal Hold and CEC-flagged "
        "dispositions, concentrating on one destination and one correspondent. Corridor "
        "concentration risk to watch."
    )
    hv = df[df["is_high_value"]]
    lv = df[~df["is_high_value"]]

    a, b, c = st.columns(3)
    a.metric("High-value tickets", f"{len(hv)}", f"{len(lv)} other")
    b.metric("% Legal Hold (HV)", f"{100*(hv['Order Status']=='Legal Hold').mean():.1f}%" if len(hv) else "—",
             f"{100*(lv['Order Status']=='Legal Hold').mean():.1f}% other" if len(lv) else "")
    c.metric("% CEC-flagged (HV)", f"{100*hv['is_compliance_flagged'].mean():.1f}%" if len(hv) else "—",
             f"{100*lv['is_compliance_flagged'].mean():.1f}% other" if len(lv) else "")

    if len(hv):
        st.markdown("##### High-value tickets detail")
        hv_table = hv[["Ticket ID", "month", "Destination Country", "Correspondent",
                       "Selling Amount", "Order Status", "subcategory", "CEC Code"]].sort_values("Selling Amount", ascending=False)
        st.dataframe(hv_table, use_container_width=True, hide_index=True)
        render_enrichment_block(hv["Ticket ID"].tolist())
    else:
        st.info("No high-value tickets in the current filter view.")


# ────────────────────────────────────────────────────────────────────────────
# §10 Digital
# ────────────────────────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown(
        "> **So what?** Digital sending locations (EW-prefix) generate the lion's share "
        "of inbound calls (overwhelmingly from EW103). Digital tickets carry a "
        "meaningfully higher compliance-flag rate than physical-store — consistent with "
        "reduced front-line ID-check friction at point of send. Biggest lever for "
        "self-service and proactive-comms work."
    )
    digital = df[df["is_digital_location"]]
    physical = df[~df["is_digital_location"]]
    rows = []
    for seg, sub in [("Digital (EW)", digital), ("Physical store", physical)]:
        rows.append({
            "Segment": seg,
            "Tickets": len(sub),
            "% of total": round(100 * len(sub) / max(len(df), 1), 1),
            "Avg selling ($)": round(sub["Selling Amount"].mean() or 0, 2),
            "Avg resolution (min)": round(sub["resolution_minutes"].mean() or 0, 1),
            "% CEC-flagged": round(100 * sub["is_compliance_flagged"].mean(), 1),
            "% Legal Hold": round(100 * (sub["Order Status"] == "Legal Hold").mean(), 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("##### Reason mix by channel-type")
    reason_mix = df.groupby(["channel_type", "subcategory"]).size().unstack(fill_value=0)
    st.bar_chart(reason_mix.T, color=PALETTE[: reason_mix.shape[0]])
    total_caption(reason_mix.values.sum())
    st.dataframe(with_totals(reason_mix), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# §11 Recommendations
# ────────────────────────────────────────────────────────────────────────────
with tabs[10]:
    st.markdown(
        "> **So what?** Prioritised actions, sized to where the data points. Each cites "
        "a number from the analysis above plus (where available) quoted ticket evidence."
    )
    rec_path = DATA_DIR / "recommendations.csv"
    if rec_path.exists():
        recs = pd.read_csv(rec_path)
        for _, r in recs.iterrows():
            priority = r["Priority"]
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority, "")
            with st.container(border=True):
                st.markdown(f"### {color} {r['Title']}")
                st.markdown(f"**Priority:** {priority}  ·  **Owner:** {r['Owner']}")
                st.markdown(f"**Finding:** {r['Finding']}")
                st.markdown(f"**Impact:** {r['Impact']}")
                st.markdown(f"**Recommendation:** {r['Recommendation']}")
    else:
        st.warning("Run `python3 build_report.py` first to generate recommendations.csv.")


# ────────────────────────────────────────────────────────────────────────────
# §12 Appendix
# ────────────────────────────────────────────────────────────────────────────
with tabs[11]:
    st.markdown(
        "> **So what?** Methodology + definitions so a reader can audit any number above."
    )
    st.markdown(
        """
**Source files** — Five Excel exports combined (Feb / Mar / Apr inbound + Mar / Apr messaging).
Feb messaging not delivered → channel-mix conclusions for Feb are inbound-only.

**APAC-origin filter** — Rows with Country ∉ {Australia, New Zealand} dropped at load (a handful
of UK / US / FR origins appear in the messaging files).

**Resolution time** = (solved − created) in minutes. Timestamps treated as UTC as-recorded.

**CEC blanks** = "not flagged"; any non-blank value = compliance-flagged.

**High-value threshold** — Selling Amount > $5,000. Messaging tickets have no Selling Amount and
are excluded from the high-value spotlight.

**Digital location** — EW-prefix Customer Sending Location.

**Repeat orders** — same Ria Order Number in more than one ticket.

**Sample size** — n = 357 across 3 months. No inferential statistics; descriptive only.
        """
    )
    st.markdown("##### Numeric summary statistics (current filtered view)")
    st.dataframe(df[["Recipient Amount", "Selling Amount", "resolution_minutes"]].describe().round(2),
                 use_container_width=True)

    st.markdown("##### Raw dataframe (current filtered view)")
    st.dataframe(df, use_container_width=True, height=400)
