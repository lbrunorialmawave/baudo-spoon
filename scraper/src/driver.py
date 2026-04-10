from __future__ import annotations

import logging
import subprocess
from typing import Generator
from contextlib import contextmanager

from seleniumbase import Driver

log = logging.getLogger(__name__)


def _safe_quit(driver: Driver) -> None:
    """Quit the driver, force-killing the Chrome process if quit hangs."""
    try:
        driver.quit()
    except Exception as exc:
        log.debug("driver.quit() raised %s — force-killing Chrome process", exc)
        try:
            pid = getattr(driver, "browser_pid", None)
            if pid:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                )
        except Exception:
            pass


@contextmanager
def get_managed_driver() -> Generator[Driver, None, None]:
    """
    Context manager that yields a robust SeleniumBase UC-mode driver.
    """
    driver = Driver(
        uc=True,
        headless=True,
        headless2=False,
        incognito=True,
    )
    try:
        yield driver
    finally:
        _safe_quit(driver)
