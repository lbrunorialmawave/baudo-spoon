/**
 * Last-resort season list, used only when `GET /quotations/seasons` is
 * unreachable. Real season selection always comes from that endpoint —
 * this only needs to stay roughly current, not exact, and deliberately
 * does not derive from the calendar date (see backend `gruppo_esperti.py`
 * for why a date-based "current season" guess is unreliable near the
 * season boundary).
 */
export const SEASON_FALLBACK_LIST: readonly number[] = [2026, 2025, 2024];
