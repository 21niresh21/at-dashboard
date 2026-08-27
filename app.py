"""AT Dashboard — Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.logging_config import setup_logging

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Must be the very first Streamlit command
st.set_page_config(
    page_title=settings.app_name,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging()

# ── Imports (after bootstrap) ─────────────────────────────────────────────────
from pages.confluence import render as render_confluence  # noqa: E402
from pages.home import render as render_home  # noqa: E402
from pages.screener import render as render_screener  # noqa: E402
from pages.signals import render as render_signals  # noqa: E402

# ── Page registry ─────────────────────────────────────────────────────────────
# Maps sidebar labels to (icon, render_function) pairs.
# Add new pages here to register them in the sidebar.
_PAGES: dict[str, tuple[str, object]] = {
    "Home": ("🏠", render_home),
    "Screener": ("", render_screener),
    "Signals": ("", render_signals),
    "Confluence": ("🎯", render_confluence),
}


# ── Sidebar navigation ────────────────────────────────────────────────────────
def sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(f"## {settings.app_name}")
        st.caption(f"Environment: `{settings.app_env.value}`")
        st.divider()

        st.markdown("#### Navigation")
        selected = st.radio(
            "Go to",
            options=list(_PAGES.keys()),
            format_func=lambda name: f"{_PAGES[name][0]}  {name}",
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown("#### Info")
        st.markdown(
            f"- **Version:** 0.1.0\n"
            f"- **Log level:** {settings.log_level}\n"
        )

    return selected


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point."""
    page_name = sidebar()
    _, render_fn = _PAGES[page_name]
    render_fn()


if __name__ == "__main__":
    main()
