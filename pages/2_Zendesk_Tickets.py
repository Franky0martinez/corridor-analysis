"""
Streamlit page: Zendesk Ticket Browser.

A direct browse view of every Zendesk ticket we've scraped into
data/ticket_enrichment.json. Sidebar filters narrow the table; clicking a
row drops you into a full-thread drill-in with first/last messages,
internal notes, and a link back to the live ticket.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from build_report import load_data, load_enrichment

ZENDESK_BASE = "https://mts-eeft.zendesk.com/agent/tickets"

st.set_page_config(
    page_title="Zendesk Ticket Browser · APAC Corridor",
    page_icon="🎫",
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


@st.cache_data
def build_enriched_table(df_full: pd.DataFrame, enrichment: dict) -> pd.DataFrame:
    """Join enrichment records onto the dataframe so filters work off one table."""
    df = df_full[df_full["Ticket ID"].astype(str).isin(enrichment.keys())].copy()
    df["_tid_str"] = df["Ticket ID"].astype(str)

    def _enr(tid: str, field: str, default=None):
        rec = enrichment.get(tid, {})
        return rec.get(field, default)

    df["enr_subject"] = df["_tid_str"].map(lambda t: _enr(t, "subject", ""))
    df["enr_requester"] = df["_tid_str"].map(lambda t: _enr(t, "requester", ""))
    df["enr_tags"] = df["_tid_str"].map(lambda t: _enr(t, "tags", []) or [])
    df["enr_first_msg"] = df["_tid_str"].map(lambda t: _enr(t, "first_customer_message", "") or "")
    df["enr_last_reply"] = df["_tid_str"].map(lambda t: _enr(t, "last_agent_reply", "") or "")
    df["enr_comment_count"] = df["_tid_str"].map(lambda t: _enr(t, "comment_count", 0) or 0)
    df["enr_tag_count"] = df["enr_tags"].apply(len)
    df["enr_has_internal"] = df["_tid_str"].map(
        lambda t: any((c.get("is_internal") for c in (_enr(t, "comments", []) or [])))
    )
    df["enr_scraped_at"] = df["_tid_str"].map(lambda t: _enr(t, "scraped_at", ""))
    # Searchable concatenation
    df["_search"] = (
        df["enr_subject"].fillna("").str.lower() + " "
        + df["enr_tags"].apply(lambda ts: " ".join(ts).lower()) + " "
        + df["enr_first_msg"].fillna("").str.lower() + " "
        + df["enr_last_reply"].fillna("").str.lower()
    )
    return df


df_full = get_df()
enrichment = get_enrichment()
df = build_enriched_table(df_full, enrichment)


# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.title("🎫 Zendesk Ticket Browser")
st.caption(
    f"Every Zendesk ticket we've scraped — {len(df):,} records with full "
    "conversation threads, tags, and metadata. Use the filters to narrow, "
    "then click a row to read the conversation."
)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

search = st.sidebar.text_input(
    "Search text",
    "",
    help="Searches subject, tags, first customer message, and last agent reply (case-insensitive substring).",
)
months = sorted(df["month"].dropna().unique().tolist())
destinations = sorted(df["Destination Country"].dropna().unique().tolist())
statuses = sorted(df["Order Status"].dropna().unique().tolist())
reasons = sorted(df["subcategory"].dropna().unique().tolist())
groups = sorted(df["Ticket group"].dropna().unique().tolist())
channels = sorted(df["channel_simple"].dropna().unique().tolist())

sel_months = st.sidebar.multiselect("Month", months, default=months)
sel_dests = st.sidebar.multiselect("Destination", destinations, default=destinations)
sel_statuses = st.sidebar.multiselect("Order Status", statuses, default=statuses)
sel_reasons = st.sidebar.multiselect("Reason (subcategory)", reasons, default=reasons)
sel_groups = st.sidebar.multiselect("Ticket group", groups, default=groups)
sel_channels = st.sidebar.multiselect("Channel", channels, default=channels)
min_comments = st.sidebar.slider("Min comments in thread", 1, 20, 1)
only_internal = st.sidebar.checkbox("Only tickets with internal notes", value=False)

filtered = df[
    df["month"].isin(sel_months)
    & df["Destination Country"].isin(sel_dests)
    & df["Order Status"].isin(sel_statuses)
    & df["subcategory"].isin(sel_reasons)
    & df["Ticket group"].isin(sel_groups)
    & df["channel_simple"].isin(sel_channels)
    & (df["enr_comment_count"] >= min_comments)
].copy()
if only_internal:
    filtered = filtered[filtered["enr_has_internal"]]
if search.strip():
    filtered = filtered[filtered["_search"].str.contains(search.strip().lower(), na=False, regex=False)]

st.sidebar.divider()
st.sidebar.metric("Tickets in view", f"{len(filtered):,}", delta=f"of {len(df):,}")

if len(filtered) == 0:
    st.warning("No tickets match the current filters.")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
# KPI strip
# ────────────────────────────────────────────────────────────────────────────
total = len(filtered)
total_comments = int(filtered["enr_comment_count"].sum())
pct_internal = 100 * filtered["enr_has_internal"].mean() if total else 0
scrape_min = filtered["enr_scraped_at"].min() or ""
scrape_max = filtered["enr_scraped_at"].max() or ""
all_tags_flat = [t for tags in filtered["enr_tags"] for t in tags]
top_tag = pd.Series(all_tags_flat).value_counts().idxmax() if all_tags_flat else "—"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tickets enriched", f"{total:,}")
c2.metric("Total comments", f"{total_comments:,}")
c3.metric("% with internal notes", f"{pct_internal:.0f}%")
c4.metric("Scrape window", f"{scrape_min[:10]} → {scrape_max[:10]}" if scrape_min else "—")
c5.metric("Most-frequent tag", top_tag)

st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────
# Browsable table
# ────────────────────────────────────────────────────────────────────────────
display = filtered.copy()
display["Subject"] = display["enr_subject"].str[:80]
display["Internal notes"] = display["enr_has_internal"].map({True: "✓", False: "—"})
display["Resolution (min)"] = display["resolution_minutes"].round(1)
display["Selling ($)"] = display["Selling Amount"].round(2)
table = display[[
    "Ticket ID", "month", "Subject",
    "Destination Country", "Order Status", "subcategory",
    "Ticket group", "channel_simple",
    "enr_comment_count", "Internal notes", "enr_tag_count",
    "Resolution (min)", "Selling ($)",
]].rename(columns={
    "month": "Month",
    "Destination Country": "Destination",
    "Order Status": "Status",
    "subcategory": "Reason",
    "Ticket group": "Group",
    "channel_simple": "Channel",
    "enr_comment_count": "Comments",
    "enr_tag_count": "Tags",
}).sort_values("Ticket ID", ascending=False)

st.caption("👆 Click a row to drill into the full conversation thread below.")
event = st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="zendesk_ticket_table",
    height=400,
)


# ────────────────────────────────────────────────────────────────────────────
# CSV export of the filtered view
# ────────────────────────────────────────────────────────────────────────────
st.download_button(
    label="📥 Download current filter as CSV (metadata only, no conversation bodies)",
    data=table.to_csv(index=False).encode(),
    file_name=f"zendesk_tickets_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)


# ────────────────────────────────────────────────────────────────────────────
# Drill-in
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if not event.selection.rows:
    st.info("Click a row above to see the full conversation thread for that ticket.")
    st.stop()

picked_idx = event.selection.rows[0]
picked_tid = str(table.iloc[picked_idx]["Ticket ID"])
row = filtered[filtered["_tid_str"] == picked_tid].iloc[0]
rec = enrichment.get(picked_tid, {})

st.markdown(f"## #{picked_tid} — {rec.get('subject', '(no subject)')}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Created", row["created"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["created"]) else "—")
m2.metric("Resolution (min)", f"{row['resolution_minutes']:.1f}" if pd.notna(row["resolution_minutes"]) else "—")
m3.metric("Status", row["Order Status"])
m4.metric("Reason", row["subcategory"])
m5.metric("Channel", row["channel_simple"])

m6, m7, m8, m9 = st.columns(4)
m6.metric("Source", row["Source"])
m7.metric("Destination", row["Destination Country"])
m8.metric("Selling ($)", f"${row['Selling Amount']:,.2f}" if pd.notna(row["Selling Amount"]) else "—")
m9.metric("Group", row["Ticket group"])

# Tags chip row
tags = rec.get("tags") or []
if tags:
    st.markdown("**Tags:** " + " · ".join(f"`{t}`" for t in tags[:25]))

# Open in Zendesk
st.link_button(
    f"↗️ Open ticket #{picked_tid} in Zendesk",
    f"{ZENDESK_BASE}/{picked_tid}",
    type="secondary",
)

# Two side-by-side message panels
fc = (rec.get("first_customer_message") or "").strip()
lr = (rec.get("last_agent_reply") or "").strip()
left, right = st.columns(2)
with left:
    st.markdown("##### First customer message")
    if fc:
        st.write(fc)
    else:
        st.caption("_No customer message captured._")
with right:
    st.markdown("##### Last agent reply")
    if lr and lr != fc:
        st.write(lr)
    else:
        st.caption("_No distinct agent reply captured._")

# Full conversation thread
comments = rec.get("comments") or []
if comments:
    st.markdown(f"##### Full conversation thread ({len(comments)} comments)")
    for c in comments:
        is_internal = c.get("is_internal", False)
        author = c.get("author") or "(unknown)"
        ts = c.get("timestamp") or ""
        body = c.get("body", "")
        with st.container(border=True):
            badge = "🟡 Internal note · " if is_internal else ""
            st.markdown(f"**{badge}{author}** · {ts}")
            st.write(body)
else:
    st.info("No conversation comments were captured for this ticket.")
