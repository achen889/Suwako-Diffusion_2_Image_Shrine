"""Small display-formatting helpers shared by the studio UI."""
from __future__ import annotations

import logging

logger = logging.getLogger("suwako")


def format_duration(seconds) -> str:
    """Format elapsed seconds as compact hours/minutes/seconds."""
    try:
        total = max(0, int(float(seconds)))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        output = f"{hours}h" if hours else ""
        if minutes or hours:
            output += f"{minutes}m"
        return output + f"{seconds}s"
    except Exception:
        logger.exception("Unable to format elapsed duration")
        return "???"
