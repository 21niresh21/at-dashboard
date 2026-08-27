"""Shared UI components for the dashboard."""

from __future__ import annotations

import streamlit as st


def page_header(title: str, description: str | None = None) -> None:
    """Render a consistent page header with optional description."""
    st.title(title)
    if description:
        st.caption(description)
    st.divider()


def metric_card(label: str, value: str | int | float, delta: str | None = None) -> None:
    """Display a single metric inside an expander-style card.

    Wraps ``st.metric`` with consistent formatting.
    """
    st.metric(label=label, value=value, delta=delta)


def empty_state(message: str, icon: str = "📭") -> None:
    """Show a centred empty-state placeholder."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:3rem 1rem; color:#888;">
            <div style="font-size:2.5rem;">{icon}</div>
            <p style="margin-top:0.5rem;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
