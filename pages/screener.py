"""Screener page — browse and sync NSE / F&O instruments from AngelOne."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import page_header
from src.database import models as db
from src.logging_config import get_logger
from src.smartapi.instruments import fetch_all_instruments

logger = get_logger(__name__)

# ── Page constants ────────────────────────────────────────────────────────────

_TAB_NSE = "NSE Cash"
_TAB_FO = "F&O Stocks"
_TAB_INDICES = "Indices"
_TAB_SYNC = "Sync & Status"

_FO_TYPE_OPTIONS = ["All", "FUTSTK", "OPTSTK", "FUTIDX", "OPTIDX"]

_PAGE_SIZE = 200


# ── Public entry point ────────────────────────────────────────────────────────


def render() -> None:
    """Render the screener page."""
    page_header(
        title="Screener",
        description="Browse NSE instruments, F&O contracts, and indices. "
        "Sync from AngelOne to refresh local data.",
    )

    # Ensure schema exists on first visit
    db.init_schema()

    tab_nse, tab_fo, tab_idx, tab_sync = st.tabs(
        [_TAB_NSE, _TAB_FO, _TAB_INDICES, _TAB_SYNC]
    )

    with tab_nse:
        _render_nse_tab()

    with tab_fo:
        _render_fo_tab()

    with tab_idx:
        _render_indices_tab()

    with tab_sync:
        _render_sync_tab()


# ── Tab renderers ─────────────────────────────────────────────────────────────


def _render_nse_tab() -> None:
    """NSE cash-segment instruments."""
    st.subheader("NSE Cash Segment")

    search = st.text_input(
        "Search by name or symbol",
        key="nse_search",
        placeholder="e.g. RELIANCE, TCS, INFY …",
    )

    records = db.get_nse_instruments(search=search, limit=_PAGE_SIZE)

    if not records:
        st.info("No NSE instruments found. Run a sync first.")
        return

    st.caption(f"Showing {len(records)} of ~10 000 NSE instruments")
    df = pd.DataFrame(records)
    _display_columns = [
        "token",
        "symbol",
        "name",
        "exchange",
        "type",
        "tick_size",
        "lot_size",
    ]
    st.dataframe(
        df[_display_columns],
        use_container_width=True,
        hide_index=True,
    )


def _render_fo_tab() -> None:
    """NSE F&O instruments with underlying + type filters."""
    st.subheader("NSE F&O Contracts")

    # Filters row
    col1, col2, col3 = st.columns(3)

    with col1:
        search = st.text_input(
            "Search symbol",
            key="fo_search",
            placeholder="e.g. RELIANCE29SEP…",
        )

    with col2:
        underlying_names = db.get_fo_underlying_names()
        underlying = st.selectbox(
            "Underlying stock",
            options=[""] + underlying_names,
            key="fo_underlying",
        )

    with col3:
        fo_type = st.selectbox(
            "Instrument type",
            options=_FO_TYPE_OPTIONS,
            key="fo_type",
        )

    type_filter = "" if fo_type == "All" else fo_type

    records = db.get_fo_instruments(
        search=search,
        underlying=underlying,
        instrument_type=type_filter,
        limit=_PAGE_SIZE,
    )

    if not records:
        st.info("No F&O instruments match your filters. Try adjusting or run a sync.")
        return

    st.caption(f"Showing {len(records)} F&O contracts")
    df = pd.DataFrame(records)
    _display_columns = [
        "token",
        "symbol",
        "name",
        "type",
        "expiry",
        "strike_price",
        "lot_size",
        "tick_size",
    ]
    st.dataframe(
        df[_display_columns],
        use_container_width=True,
        hide_index=True,
    )


def _render_indices_tab() -> None:
    """Index instruments across all exchanges."""
    st.subheader("Indices")

    search = st.text_input(
        "Search index name",
        key="idx_search",
        placeholder="e.g. NIFTY, BANKNIFTY, FINNIFTY …",
    )

    records = db.get_indices(search=search, limit=_PAGE_SIZE)

    if not records:
        st.info("No indices found. Run a sync first.")
        return

    st.caption(f"Showing {len(records)} indices")
    df = pd.DataFrame(records)
    _display_columns = [
        "token",
        "symbol",
        "name",
        "exchange",
        "type",
        "tick_size",
    ]
    st.dataframe(
        df[_display_columns],
        use_container_width=True,
        hide_index=True,
    )


def _render_sync_tab() -> None:
    """Sync controls and status information."""
    st.subheader("Data Sync")

    # ── Sync button ───────────────────────────────────────────────────────────
    st.markdown(
        "Sync downloads the full instrument master list from AngelOne's CDN "
        "and refreshes the local database. This may take a few seconds."
    )

    if st.button("Sync from AngelOne", type="primary", use_container_width=False):
        _perform_sync()

    st.divider()

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.subheader("Database Summary")

    counts = db.get_instrument_counts()
    last_sync = db.get_last_sync()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NSE Instruments", counts.get("NSE", 0))
    with col2:
        st.metric("NFO Instruments", counts.get("NFO", 0))
    with col3:
        total = sum(counts.values())
        st.metric("Total Instruments", total)

    if last_sync:
        st.divider()
        st.markdown("**Last sync**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Status:** `{last_sync['status']}`")
        with col2:
            st.markdown(f"**Records:** {last_sync['record_count']:,}")
        with col3:
            st.markdown(f"**Completed:** {last_sync.get('completed_at', 'N/A')}")

    # ── Sync history ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Sync History")

    history = db.get_sync_history(limit=10)
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No sync history yet.")


# ── Sync logic ────────────────────────────────────────────────────────────────


def _perform_sync() -> None:
    """Execute the full sync pipeline with progress feedback."""
    progress = st.progress(0, text="Fetching instrument data from AngelOne …")

    try:
        progress.progress(10, text="Downloading master script …")
        records = fetch_all_instruments()

        progress.progress(50, text=f"Received {len(records):,} instruments. Writing to database …")
        count = db.sync_instruments(records)

        progress.progress(100, text=f"Sync complete — {count:,} instruments stored.")
        st.success(f"Successfully synced {count:,} instruments.")
        logger.info("Manual sync completed: %d instruments.", count)

    except Exception as exc:
        progress.progress(100, text="Sync failed.")
        st.error(f"Sync failed: {exc}")
        logger.exception("Manual sync failed.")
