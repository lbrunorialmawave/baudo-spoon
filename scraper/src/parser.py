from __future__ import annotations

import logging
from typing import Any

from selenium.webdriver.common.by import By

log = logging.getLogger(__name__)


def parse_match_link(link_element: Any, match_url: str) -> dict[str, Any] | None:
    """
    Parses a single match link element to extract match data.
    """
    try:
        match_text = link_element.text
        if not match_text:
            return None

        lines = [line.strip() for line in match_text.split("\n") if line.strip()]

        score = None
        status = None
        match_time = None
        teams: list[str] = []

        for line in lines:
            if line in ("AM", "PM"):
                if match_time:
                    match_time = f"{match_time} {line}"
                continue
            if ":" in line and len(line) <= 5 and all(c.isdigit() or c == ":" for c in line):
                match_time = line
            elif " - " in line and any(c.isdigit() for c in line):
                score = line
            elif (
                line in ("FT", "Live", "HT", "Postponed", "Cancelled", "AET", "PEN")
                or line.endswith("'")
                or (len(line) <= 5 and "+" in line and any(c.isdigit() for c in line))
                or (line.isdigit() and len(line) <= 3)
            ):
                status = line
            elif (
                line not in ("FT", "Live", "HT", "Postponed", "Cancelled", "AET", "PEN", "AM", "PM")
                and " - " not in line
                and not (":" in line and len(line) <= 5)
            ):
                teams.append(line)

        if len(teams) >= 2:
            return {
                "home": teams[0],
                "away": teams[1],
                "score": score if score else (match_time if match_time else "N/A"),
                "status": status if status else ("Upcoming" if match_time else "N/A"),
                "url": match_url,
            }
    except Exception as exc:
        log.debug("Failed to parse match link text: %s", exc)

    return None


def calculate_match_points(home_goals: int, away_goals: int) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if away_goals > home_goals:
        return 0, 3
    return 1, 1


def parse_score(score_str: str | None) -> tuple[int, int]:
    if score_str and " - " in score_str:
        try:
            parts = score_str.split(" - ")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    return 0, 0


def create_team_rows(match: dict[str, Any], round_num: int) -> tuple[dict[str, Any], dict[str, Any]]:
    base_info = {
        "Date": match.get("date", "N/A"),
        "Round": round_num,
        "Match": f"{match['home']} - {match['away']}",
        "Score": match["score"],
        "Status": match["status"],
        "Url": match["url"],
    }

    home_goals, away_goals = parse_score(match["score"])
    home_points, away_points = calculate_match_points(home_goals, away_goals)

    home_row = base_info.copy()
    home_row.update({
        "Team": match["home"],
        "Side": "Home",
        "Opponent": match["away"],
        "Goal scored": home_goals,
        "Goal conceded": away_goals,
        "points": home_points,
    })

    away_row = base_info.copy()
    away_row.update({
        "Team": match["away"],
        "Side": "Away",
        "Opponent": match["home"],
        "Goal scored": away_goals,
        "Goal conceded": home_goals,
        "points": away_points,
    })

    return home_row, away_row


def extract_possession(driver: Any, stats_data: dict[str, Any]) -> None:
    try:
        possession_div = driver.find_element(By.CSS_SELECTOR, "div.css-1xzakdb-PossessionDiv")
        possession_spans = possession_div.find_elements(By.TAG_NAME, "span")

        if len(possession_spans) >= 2:
            home_possession = possession_spans[0].text.strip()
            away_possession = possession_spans[1].text.strip()

            if "Top stats" not in stats_data:
                stats_data["Top stats"] = {}

            stats_data["Top stats"]["Ball possession"] = [home_possession, away_possession]
    except Exception:
        pass


def extract_stat_sections(driver: Any, stats_data: dict[str, Any]) -> None:
    stat_containers = driver.find_elements(By.CSS_SELECTOR, "ul.css-1pxkecz-StatGroupContainer")

    for container in stat_containers:
        current_section = "Top stats"
        try:
            header = container.find_element(By.CSS_SELECTOR, "header h2")
            if header:
                current_section = header.text.strip()
        except Exception:
            pass

        items = container.find_elements(By.CSS_SELECTOR, "li")

        for item in items:
            try:
                if _is_section_header(item):
                    new_section = item.text.strip()
                    if new_section:
                        current_section = new_section
                        if current_section not in stats_data:
                            stats_data[current_section] = {}
                    continue

                stat_name, home_val, away_val = _parse_stat_row(item)
                if stat_name:
                    if current_section not in stats_data:
                        stats_data[current_section] = {}
                    stats_data[current_section][stat_name] = [home_val, away_val]
            except Exception:
                continue


def _is_section_header(item: Any) -> bool:
    try:
        item.find_element(By.CSS_SELECTOR, "[class*='StatBox'], [class*='StatValue']")
        return False
    except Exception:
        return bool(item.text.strip())


def _parse_stat_row(item: Any) -> tuple[str | None, str | None, str | None]:
    text = item.text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) >= 3:
        # Format: HomeValue \n StatName \n AwayValue
        return lines[1], lines[0], lines[2]

    return None, None, None
