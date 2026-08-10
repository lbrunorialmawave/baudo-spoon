"""Elimina i refresh token scaduti/revocati oltre la finestra di retention.

Standalone: non importa l'app FastAPI, si connette direttamente al DB via
psycopg2. Pensato per girare via cron (GitHub Actions, vedi
``.github/workflows/refresh-token-cleanup.yml``) o manualmente.

Retention: le righe revocate/scadute NON vengono eliminate subito. Si tiene
una finestra (default 7 giorni) per poter fare indagini di sicurezza post-
incidente (es. correlare un ``reuse_detected`` con i log applicativi) prima
che la riga sparisca.

Uso:
    python -m scripts.cleanup_refresh_tokens --retention-days 7
    python -m scripts.cleanup_refresh_tokens --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import psycopg2

log = logging.getLogger("cleanup_refresh_tokens")

_DELETE_SQL = """
    DELETE FROM refresh_tokens
    WHERE
        -- Revocati (rotated / reuse_detected / logout / logout_all) da più
        -- della finestra di retention.
        (revoked_at IS NOT NULL AND revoked_at < NOW() - MAKE_INTERVAL(days => %(retention_days)s))
        OR
        -- Mai revocati esplicitamente ma scaduti naturalmente da più della
        -- finestra di retention (es. l'utente non è mai tornato ad usarli).
        (revoked_at IS NULL AND expires_at < NOW() - MAKE_INTERVAL(days => %(retention_days)s))
"""

_COUNT_SQL = _DELETE_SQL.replace("DELETE FROM refresh_tokens", "SELECT COUNT(*) FROM refresh_tokens")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("REFRESH_TOKEN_RETENTION_DAYS", "7")),
        help="Giorni da tenere i token revocati/scaduti prima di eliminarli (default 7).",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("API_DATABASE_URL"),
        help="DSN Postgres (sync). Default: env API_DATABASE_URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Conta le righe candidate senza eliminarle.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.database_url:
        log.error("Missing database URL: set API_DATABASE_URL or pass --database-url")
        return 2

    if args.retention_days < 1:
        log.error("--retention-days must be >= 1 (got %s)", args.retention_days)
        return 2

    params = {"retention_days": args.retention_days}

    try:
        conn = psycopg2.connect(args.database_url)
    except psycopg2.OperationalError as exc:
        log.error("Could not connect to the database: %s", exc)
        return 1

    try:
        with conn:
            with conn.cursor() as cur:
                if args.dry_run:
                    cur.execute(_COUNT_SQL, params)
                    (count,) = cur.fetchone()
                    log.info(
                        "[dry-run] %d refresh token row(s) would be deleted (retention=%dd)",
                        count,
                        args.retention_days,
                    )
                else:
                    cur.execute(_DELETE_SQL, params)
                    log.info(
                        "Deleted %d refresh token row(s) (retention=%dd)",
                        cur.rowcount,
                        args.retention_days,
                    )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
