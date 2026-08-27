"""Home page — application landing screen."""

from __future__ import annotations

import streamlit as st

from src.components.ui import page_header
from src.logging_config import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the home page content."""
    page_header(
        title="AT Dashboard",
        description="Welcome to the AT trading dashboard. Use the sidebar to navigate.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Overview")
        st.markdown(
            "Monitor portfolio performance, key metrics, and real-time market data "
            "at a glance."
        )

    with col2:
        st.markdown("### 📈 Strategy")
        st.markdown(
            "Configure and review active trading strategies, signal history, "
            "and back-test results."
        )

    with col3:
        st.markdown("### 🛒 Orders")
        st.markdown(
            "Track open orders, execution history, and order-level analytics."
        )

    st.divider()

    st.info(
        "This is the home screen. Additional pages are available in the sidebar."
    )
    logger.info("Home page rendered")
