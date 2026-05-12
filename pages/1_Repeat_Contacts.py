"""
Streamlit page: Repeat-Contact Orders.

Order-level rollup + drill-in into the conversation threads (where enriched).
Loads the same dataframe as the main app via build_report.load_data().
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from build_report import (
    DATA_DIR,
    ENRICHMENT_PATH,
    NON_TERMINAL_STATUSES,
    PALETTE,
    load_data,
    load_enrichment,
)

st.set_page_config(
    page_title="Repeat-Contact Orders · APAC Corridor",
    page_icon="🔁",
    layout="wide",
)


# ────────────────────────────────────────────────────────────────────────────
# Data
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
# Header + intro
# ────────────────────────────────────────────────────────────────────────────
st.title("🔁 Repeat-Contact Orders")
st.markdown(
    """
> **So what?** When the same order generates multiple tickets, the customer is calling
> back because their issue wasn't resolved on first contact. This is "repeat-contact
> friction" — a quality-of-service signal that compounds handle-cost and erodes trust.
"""
)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
months = sorted(df_full["month"].dropna().unique().tolist())
reason_categories = sorted(df_full["category"].dropna().unique().tolist())
reason_subcategories = sorted(df_full["subcategory"].dropna().unique().tolist())
destinations = sorted(df_full["Destination Country"].dropna().unique().tolist())

sel_months = st.sidebar.multiselect("Month", months, default=months)
sel_dests = st.sidebar.multiselect("Destination country", destinations, default=destinations)
sel_reason_cats = st.sidebar.multiselect(
    "Reason category", reason_categories, default=reason_categories
)
sel_reason_subs = st.sidebar.multiselect(
    "Reason subcategory", reason_subcategories, default=reason_subcategories
)
min_repeats = st.sidebar.slider("Min ticket count per order", min_value=2, max_value=10, value=2)
sort_by = st.sidebar.selectbox(
    "Sort rollup by",
    ["Ticket count (worst first)", "Total selling amount", "Most recent contact", "Order number"],
)

df = df_full[
    df_full["month"].isin(sel_months)
    & df_full["Destination Country"].isin(sel_dests)
    & df_full["category"].isin(sel_reason_cats)
    & df_full["subcategory"].isin(sel_reason_subs)
].copy()


# ────────────────────────────────────────────────────────────────────────────
# Compute order-level rollup
# ────────────────────────────────────────────────────────────────────────────
counts = df.groupby("Ria Order Number").size().rename("ticket_count")
repeat_orders = counts[counts >= min_repeats].index.tolist()
df_rep = df[df["Ria Order Number"].isin(repeat_orders)].copy()

if len(df_rep) == 0:
    st.warning(
        f"No orders with at least {min_repeats} tickets in the current filter window. "
        f"Loosen the filters or lower the min-repeats slider."
    )
    st.stop()

rollup = (
    df_rep.groupby("Ria Order Number")
    .agg(
        Tickets=("Ticket ID", "count"),
        UniqueChannels=("data_source", "nunique"),
        TotalSelling=("Selling Amount", "sum"),
        AvgResolutionMin=("resolution_minutes", "mean"),
        FirstContact=("created", "min"),
        LastContact=("created", "max"),
        CurrentStatus=("Order Status", lambda s: s.mode().iloc[0] if len(s) else ""),
        StatusesSeen=("Order Status", lambda s: ", ".join(sorted(s.dropna().unique()))),
        DestinationCountry=("Destination Country", "first"),
        ReasonMix=("subcategory", lambda s: ", ".join(s.value_counts().head(3).index.tolist())),
        EnrichedTickets=("Ticket ID", lambda s: sum(1 for t in s if str(t) in enrichment)),
    )
)
rollup["Months"] = (rollup["LastContact"] - rollup["FirstContact"]).dt.days
rollup["AvgResolutionMin"] = rollup["AvgResolutionMin"].round(1)
rollup["TotalSelling"] = rollup["TotalSelling"].round(2)

sort_col = {
    "Ticket count (worst first)": "Tickets",
    "Total selling amount": "TotalSelling",
    "Most recent contact": "LastContact",
    "Order number": "Ria Order Number",
}[sort_by]
ascending = (sort_by == "Order number")
rollup = rollup.sort_values(sort_col, ascending=ascending)


# ────────────────────────────────────────────────────────────────────────────
# KPI row
# ────────────────────────────────────────────────────────────────────────────
total_orders = len(rollup)
total_tickets = int(rollup["Tickets"].sum())
total_value = float(rollup["TotalSelling"].sum())
enriched_orders = int((rollup["EnrichedTickets"] > 0).sum())
worst_order = rollup.index[0] if len(rollup) else "—"
worst_n = int(rollup["Tickets"].max()) if len(rollup) else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Repeat-contact orders", f"{total_orders:,}")
c2.metric("Total repeat tickets", f"{total_tickets:,}")
c3.metric("Total selling value", f"${total_value:,.0f}")
c4.metric("Orders with quoted evidence", f"{enriched_orders}/{total_orders}")
c5.metric("Worst offender", f"{worst_order}", f"{worst_n} tickets")

st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────
# Rollup table
# ────────────────────────────────────────────────────────────────────────────
st.markdown("### Order-level rollup")
st.caption("Every order with multiple tickets in the current filter. Click a row's order number, then scroll down to drill in.")

display_rollup = rollup.copy().reset_index()
display_rollup["FirstContact"] = display_rollup["FirstContact"].dt.strftime("%Y-%m-%d")
display_rollup["LastContact"] = display_rollup["LastContact"].dt.strftime("%Y-%m-%d")
display_rollup = display_rollup.rename(columns={
    "Ria Order Number": "Order",
    "Tickets": "Tix",
    "UniqueChannels": "Channels",
    "TotalSelling": "Total selling ($)",
    "AvgResolutionMin": "Avg res (min)",
    "FirstContact": "First",
    "LastContact": "Last",
    "CurrentStatus": "Mode status",
    "StatusesSeen": "All statuses seen",
    "DestinationCountry": "Destination",
    "ReasonMix": "Top reasons",
    "EnrichedTickets": "Enriched",
    "Months": "Span (days)",
})

# Append a totals row
totals = {
    "Order": "TOTAL",
    "Tix": int(display_rollup["Tix"].sum()),
    "Channels": "",
    "Total selling ($)": round(display_rollup["Total selling ($)"].sum(), 2),
    "Avg res (min)": round(df_rep["resolution_minutes"].mean(), 1),
    "First": "",
    "Last": "",
    "Mode status": "",
    "All statuses seen": "",
    "Destination": "",
    "Top reasons": "",
    "Enriched": int(display_rollup["Enriched"].sum()),
    "Span (days)": "",
}
display_rollup = pd.concat([display_rollup, pd.DataFrame([totals])], ignore_index=True)

st.dataframe(
    display_rollup,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Total selling ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Avg res (min)": st.column_config.NumberColumn(format="%.1f"),
    },
)

# Reason mix chart across repeat-contact tickets
st.markdown("### Reason mix across repeat-contact tickets")
reason_chart = df_rep["subcategory"].value_counts()
st.bar_chart(reason_chart, color=PALETTE[0])
st.caption(f"**Total: {int(reason_chart.sum()):,} repeat-contact tickets**")


# ────────────────────────────────────────────────────────────────────────────
# Drill-in
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Drill into a specific order")

# Build order-picker options: "AU1157296364 — 7 tickets — $7,000"
def _order_label(order_num: str) -> str:
    row = rollup.loc[order_num]
    return f"{order_num} — {int(row['Tickets'])} tickets — ${row['TotalSelling']:,.0f} — {row['DestinationCountry']}"

picker_options = ["—"] + [_order_label(o) for o in rollup.index.tolist()]
picked = st.selectbox("Pick an order to inspect every contact", picker_options, index=1 if len(picker_options) > 1 else 0)

if picked != "—":
    chosen_order = picked.split(" — ")[0]
    tickets = df_rep[df_rep["Ria Order Number"] == chosen_order].sort_values("created")

    # Order-level summary card
    st.markdown(f"#### Order {chosen_order}")
    sa, sb, sc, sd = st.columns(4)
    sa.metric("Tickets", len(tickets))
    sb.metric("Selling amount", f"${tickets['Selling Amount'].iloc[0]:,.2f}" if pd.notna(tickets['Selling Amount'].iloc[0]) else "—")
    sc.metric("Destination", tickets["Destination Country"].iloc[0])
    sd.metric("Final status", tickets["Order Status"].iloc[-1])

    # Timeline table
    st.markdown("##### Contact timeline")
    timeline = tickets[[
        "Ticket ID", "created", "channel_simple", "subcategory",
        "Order Status", "Ticket group", "resolution_minutes", "Source",
    ]].copy()
    timeline["created"] = timeline["created"].dt.strftime("%Y-%m-%d %H:%M")
    timeline["resolution_minutes"] = timeline["resolution_minutes"].round(1)
    timeline = timeline.rename(columns={
        "created": "Created (UTC)",
        "channel_simple": "Channel",
        "subcategory": "Reason",
        "Order Status": "Status at pull",
        "Ticket group": "Group",
        "resolution_minutes": "Resolution (min)",
    })
    st.dataframe(timeline, use_container_width=True, hide_index=True)

    # Quoted evidence per ticket
    st.markdown("##### Conversation evidence (where enriched)")
    any_enrichment = False
    for _, row in tickets.iterrows():
        tid = str(int(row["Ticket ID"]))
        if tid not in enrichment:
            continue
        any_enrichment = True
        rec = enrichment[tid]
        with st.expander(
            f"Ticket #{tid} · {row['created'].strftime('%Y-%m-%d %H:%M')} · "
            f"{row['channel_simple']} · {row['Order Status']} — {rec.get('subject', '')}",
            expanded=False,
        ):
            tags = rec.get("tags") or []
            if tags:
                st.caption("Tags: " + " · ".join(tags[:15]))
            first = (rec.get("first_customer_message") or "").strip()
            last = (rec.get("last_agent_reply") or "").strip()
            if first:
                st.markdown(f"**First customer message:** _{first[:600]}_")
            if last and last != first:
                st.markdown(f"**Last agent reply:** _{last[:600]}_")
            comments = rec.get("comments") or []
            if comments:
                with st.expander(f"View full thread ({len(comments)} comments)"):
                    for c in comments[:15]:
                        st.markdown(
                            f"**{c.get('author') or '(unknown)'}** · {c.get('timestamp') or ''}  \n"
                            f"{c.get('body', '')[:800]}"
                        )
                        st.divider()
    if not any_enrichment:
        st.info(
            "💬 None of this order's tickets have Zendesk enrichment yet. "
            "Re-run the scrape from build_report.py / the priority-subset script to capture them."
        )
