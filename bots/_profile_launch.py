"""Utilities for launching Playwright with optional persistent profiles.

This mirrors the minimal helper that existed in the earlier Codex-powered
prototype so that the executor can opt into Chromium profiles for specific
domains (e.g. Notion). The helpers return both the Playwright controller and
objects required for shutdown so callers can ensure resources are released.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright


def launch_persistent(
    start_url: Optional[str],
    profile_dir: str,
    *,
    headless: bool = False,
) -> Tuple[Playwright, BrowserContext, Page]:
    """Launch a persistent Chromium context backed by ``profile_dir``.

    Parameters
    ----------
    start_url:
        Optional URL to navigate to immediately after launch. When omitted the
        caller can decide what to load next.
    profile_dir:
        Directory that stores Chromium profile state (cookies, localStorage,
        session data, etc.). The directory is created automatically when
        missing so repeated runs can reuse the same profile.
    headless:
        Whether to launch Chromium in headless mode. Default keeps the UI
        visible which is helpful for iterative debugging.
    """

    profile_path = Path(profile_dir)
    profile_path.mkdir(parents=True, exist_ok=True)

    playwright = sync_playwright().start()
    context = playwright.chromium.launch_persistent_context(
        str(profile_path),
        headless=headless,
    )

    if context.pages:
        page = context.pages[0]
    else:
        page = context.new_page()

    if start_url:
        try:
            page.goto(start_url, wait_until="load")
        except Exception:
            # Allow the caller to recover; a later retry can be attempted if
            # navigation fails because of offline state or auth prompts.
            pass

    return playwright, context, page


def shutdown(playwright: Optional[Playwright], context: Optional[BrowserContext]) -> None:
    """Gracefully dispose of Playwright resources used by ``launch_persistent``."""

    try:
        if context:
            context.close()
    finally:
        if playwright:
            playwright.stop()

