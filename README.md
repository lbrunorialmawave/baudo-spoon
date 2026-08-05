# baudo-spoon

> Fanta-football intelligence platform — data scraping, ML clustering & prediction, and REST API.

[![CI](https://github.com/lbrunori/baudo-spoon/actions/workflows/ci.yml/badge.svg)](https://github.com/lbrunori/baudo-spoon/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Services](#services)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Overview

**baudo-spoon** is the backend monorepo for a fanta-football analytics platform. It scrapes player statistics from public sources, trains role-partitioned ML models for fantavoto prediction, runs K-Means clustering with PCA, and exposes all results through a versioned REST API backed by PostgreSQL and Redis.

---

## Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────────┐     ┌──────────┐
│  Scraper   │────▶│ PostgreSQL │◀────│  ML Pipeline   │────▶│Artifacts │
│ (Selenium) │     │  (fbref)   │     │(XGBoost+KMeans)│     │ (joblib) │
└────────────┘     └────────────┘     └────────────────┘     └────┬─────┘
                                                                   │
                        ┌──────────┐                               │
                        │  Redis   │◀──────────────────────────────┤
                        │ (cache)  │                               │
                        └──────────┘                               │
                                                                   ▼
                                                          ┌────────────────┐
                                                          │  FastAPI (API) │
                                                          │  /api/v1       │
                                                          └────────────────┘
```

---

## Services

| Service    | Tech                          | Description                                              |
|------------|-------------------------------|----------------------------------------------------------|
| `scraper`  | Python, Selenium, SQLAlchemy  | Scrapes match statistics and player profiles from FBref  |
| `ml`       | scikit-learn, XGBoost, pandas | Role-partitioned regression + K-Means/PCA clustering     |
| `api`      | FastAPI, asyncpg, Redis       | REST API serving predictions and intelligence endpoints  |
| `db`       | PostgreSQL 16                 | Persistent storage for raw stats and player profiles     |
| —          | `manual_resolutions` (table)  | Permanent history of manually resolved Fantacalcio↔FotMob ID mappings. Survives re-mapping runs and is reused as Pass 0 in the matching pipeline. |

---

## Prerequisites

- [Docker](https://www.docker.com/) >= 24
- [Docker Compose](https://docs.docker.com/compose/) >= 2.20
- Python 3.11+ (for local development only)

---

## Getting Started

### 1. Clone and configure

```bash
git clone https://github.com/lbrunori/baudo-spoon.git
cd baudo-spoon
cp .env.example .env
# Edit .env and set POSTGRES_PASSWORD and API_KEY_SECRET
```

### 2. Start core infrastructure

```bash
docker compose up -d
```

This starts `db`, `redis`, and `api`. The API is available at `http://localhost:8000/api/v1/docs`.

### 3. Run the scraper (first time)

```bash
docker compose --profile scraper run --rm scraper --leagues "Serie A"
```

### 4. Train ML models

```bash
docker compose --profile ml run --rm ml
```

Artifacts are written to the `ml_artifacts` Docker volume and read by the API on startup.

### 5. Season onboarding (new listino / players new to Serie A)

Each step below is available both as an **Admin panel button** (Scraper Manager card) and as a **manual command**, in case the panel/Docker isn't reachable — all of them just call `POST /admin/scrape/*` under the hood, which require an admin API key (`X-API-Key` header or the panel's logged-in session). `$API_URL` below is the API's base URL including the `/api/v1` prefix, e.g. `http://localhost:8000/api/v1`.

1. **Import the new season's quotazioni.** Drop `Quotazioni_Fantacalcio_Stagione_<YYYY_YY>.xlsx` into `./quotazioni/` (auto-discovered by filename, no code change needed), then either click **"Import Quotazioni"** in the admin panel, or:
   ```bash
   curl -X POST -H "X-API-Key: $API_KEY" "$API_URL/admin/scrape/quotazioni"
   # or, with direct DB/file access (no running api container needed):
   python -m ml.data.import_quotations --quotazioni-dir ./quotazioni --source listone_fantagazzetta
   ```

2. **Fetch career stats for players new to Serie A.** After importing, some listino players (transfers from other leagues) will have no Serie A history yet — MANTRA would otherwise score them on a blank role-median guess (see `ml/mantra/pilastro1.py`'s neo-arrivo handling). Click **"Storico Giocatori Esteri"** in the admin panel (checkbox `force=true` to re-fetch players that already have some data), or:
   ```bash
   curl -X POST -H "X-API-Key: $API_KEY" "$API_URL/admin/scrape/foreign-stats?force=false"
   ```
   This finds listino players with no Serie A history and fetches each one's most recent season (any league) directly from FotMob — one lightweight request per player, not a bulk league scrape. See `scraper/src/player_career_scraper.py` for how the fallback data is derived (appearances/rating from FotMob's career-history payload; minutes and per-90 rates are estimated from appearances, not exact — every persisted row is coarser than a real per-90 scrape).

3. **Run MANTRA for the season.** Admin panel "Esegui MANTRA" button, or:
   ```bash
   curl -X POST -H "X-API-Key: $API_KEY" "$API_URL/mantra/run?season_start=<year>"
   ```
   `season_start` can be omitted — it resolves to the latest season present in `player_quotations`. Players still missing Serie A data after step 2 (e.g. their foreign league isn't in `LEAGUE_CATALOG`, or the FotMob fetch failed) fall back to `ml/mantra/pilastro1.py`'s role-median estimate as before — check each player's `stats_from_prior_season` / `stats_from_foreign_league` flags in the MANTRA output JSON to see which tier was actually used.

4. **Verify coverage**: `GET /admin/data-health` reports row counts and seasons per source; `GET /admin/scrape/status` lists each scraper's configurable params and expected frequency.

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and adjust:

| Variable                      | Default     | Description                                   |
|-------------------------------|-------------|-----------------------------------------------|
| `POSTGRES_PASSWORD`           | —           | **Required.** PostgreSQL password             |
| `API_PORT`                    | `8000`      | Exposed port for the API                      |
| `API_KEY_SECRET`              | `""`        | Bearer token for `/intelligence` endpoints    |
| `API_CACHE_TTL_SECONDS`       | `3600`      | Redis TTL for ML artifact responses           |
| `API_RATE_LIMIT_REQUESTS`     | `60`        | Rate limit window request count               |
| `API_RATE_LIMIT_WINDOW_SECONDS` | `60`      | Rate limit sliding window duration            |
| `LOG_LEVEL`                   | `INFO`      | Log verbosity (`DEBUG`, `INFO`, `WARNING`)    |
| `ML_RANDOM_SEED`              | `42`        | Reproducibility seed for ML pipeline          |

---

## Development

### API (local)

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

### ML pipeline (local)

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --league "Serie A"
```

### Scraper (local)

```bash
cd scraper
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --leagues "Serie A"
```

---

## Testing

```bash
# ML tests
cd ml
pytest tests/ -v

# API tests (requires running db + redis)
cd api
pytest tests/ -v
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(ml): add role-partitioned XGBoost regression
fix(api): handle missing Redis gracefully on startup
chore(deps): bump fastapi to 0.115
```
