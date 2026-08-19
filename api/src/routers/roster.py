"""Roster import endpoints — La Mia Squadra (runtime-only rose).

Flow: Upload Excel → parse → match against listone → RosterContext in
process-local store → return context_id + team cards.  No rose rows are
written to PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ml.roster_import import (
    CatalogPlayer,
    build_roster_context,
    parse_bytes,
)
from ml.roster_import.store import (
    DEFAULT_TTL_SECONDS,
    RosterContextStore,
    get_default_store,
)

from ..data_repository import DataRepository
from ..deps import get_db, rate_limit, require_role
from ..schemas import (
    RosterClaimRequestSchema,
    RosterClaimResponseSchema,
    RosterDetailResponseSchema,
    RosterImportResponseSchema,
    RosterMatchQualitySchema,
    RosterPlayerSchema,
    RosterTeamCardSchema,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/roster", tags=["roster"])

# Accept common Fantagazzetta export content types
_ALLOWED_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/octet-stream",  # some browsers
    "application/zip",  # xlsx is a zip
}
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MiB


def _get_repo(request: Request) -> DataRepository:
    repo: DataRepository | None = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="Data repository not initialised")
    return repo


def _get_store(request: Request) -> RosterContextStore:
    store = getattr(request.app.state, "roster_store", None)
    if store is None:
        store = get_default_store()
        request.app.state.roster_store = store
    return store


def _rows_to_catalog(rows: list[dict]) -> list[CatalogPlayer]:
    return [
        CatalogPlayer(
            fantacalcio_id=int(r["fantacalcio_id"]),
            name=str(r["name"]),
            team=str(r.get("team") or ""),
            role_classic=r.get("role_classic"),
            roles_mantra=tuple(r.get("roles_mantra") or ()),
        )
        for r in rows
    ]


def _team_cards(ctx) -> list[RosterTeamCardSchema]:
    cards: list[RosterTeamCardSchema] = []
    for div in ctx.divisions:
        for t in div.teams:
            cards.append(
                RosterTeamCardSchema(
                    sheet_name=div.sheet_name,
                    team_name=t.team_name,
                    player_count=len(t.players),
                    total_spent=t.total_spent,
                    is_empty=t.is_empty,
                    match_rate=round(t.match_rate, 4),
                )
            )
    return cards


def _quality_schema(ctx) -> RosterMatchQualitySchema:
    q = ctx.quality
    return RosterMatchQualitySchema(
        total_players=q.total_players,
        auto=q.auto,
        provisional=q.provisional,
        unmatched=q.unmatched,
        match_rate=round(q.match_rate, 4),
    )


# ── POST /roster/import ──────────────────────────────────────────────────────


@router.post(
    "/import",
    response_model=RosterImportResponseSchema,
    response_class=ORJSONResponse,
    summary="Import Fantagazzetta rose Excel",
    description=(
        "Upload a multi-division rose export. The file is parsed and matched "
        "against the official listone in memory. A short-lived ``contextId`` "
        "is returned; no current roster is persisted to the database."
    ),
    responses={
        200: {"description": "Import + match completed"},
        400: {"description": "Invalid or unreadable file"},
        413: {"description": "File too large"},
    },
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def import_roster(
    request: Request,
    file: UploadFile = File(..., description="Excel .xlsx rose export"),
    season_start: Optional[int] = Query(
        None, ge=1990, le=2100, description="Listone season (default: current)"
    ),
    db: AsyncSession = Depends(get_db),
) -> RosterImportResponseSchema:
    # Size / type guards
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        # Still allow if filename ends with .xlsx (some clients send wrong MIME)
        fname = (file.filename or "").lower()
        if not fname.endswith((".xlsx", ".xlsm")):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo file non supportato: {content_type or 'sconosciuto'}. "
                "Carica un export Excel (.xlsx) di Fantagazzetta.",
            )

    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File troppo grande (max {_MAX_UPLOAD_BYTES // (1024*1024)} MiB)",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="File vuoto")

    filename = file.filename or "upload.xlsx"

    try:
        workbook = parse_bytes(raw, source_filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    repo = _get_repo(request)
    catalog_rows = await repo.load_catalog_for_roster_matching(db, season_start=season_start)
    if not catalog_rows:
        raise HTTPException(
            status_code=503,
            detail="Listone non disponibile per la stagione richiesta. "
            "Importa prima le quotazioni ufficiali.",
        )
    catalog = _rows_to_catalog(catalog_rows)

    ctx = build_roster_context(workbook, catalog)
    store = _get_store(request)
    store.put(ctx)

    return RosterImportResponseSchema(
        context_id=ctx.context_id,
        source_filename=ctx.source_filename,
        quality=_quality_schema(ctx),
        teams=_team_cards(ctx),
        divisions=[d.sheet_name for d in ctx.divisions],
        expires_in_seconds=store.ttl_seconds,
    )


# ── POST /roster/claim ───────────────────────────────────────────────────────


@router.post(
    "/claim",
    response_model=RosterClaimResponseSchema,
    response_class=ORJSONResponse,
    summary="Claim a team as own after import",
    description=(
        "Select which fantasy team in the imported context is the user's. "
        "Updates the in-memory RosterContext only."
    ),
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def claim_team(
    request: Request,
    body: RosterClaimRequestSchema,
) -> RosterClaimResponseSchema:
    store = _get_store(request)
    ctx = store.get(body.context_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Context non trovato o scaduto. Ricarica il file Excel.",
        )

    try:
        ctx2 = ctx.with_user_team(body.sheet_name, body.team_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store.update(ctx2)
    team = ctx2.get_user_team()
    assert team is not None

    return RosterClaimResponseSchema(
        context_id=ctx2.context_id,
        user_team_key=ctx2.user_team_key or "",
        team_name=team.team_name,
        sheet_name=body.sheet_name,
        player_count=len(team.players),
        total_spent=team.total_spent,
        match_rate=round(team.match_rate, 4),
    )


# ── GET /roster/context/{context_id}/teams ───────────────────────────────────


@router.get(
    "/context/{context_id}/teams",
    response_class=ORJSONResponse,
    summary="List teams in a roster context",
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def list_context_teams(
    request: Request,
    context_id: str,
    division: Optional[str] = Query(None, description="Filter by sheet name"),
    include_empty: bool = Query(False),
) -> ORJSONResponse:
    store = _get_store(request)
    ctx = store.get(context_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Context non trovato o scaduto. Ricarica il file Excel.",
        )
    cards = []
    for div in ctx.divisions:
        if division and div.sheet_name != division:
            continue
        for t in div.teams:
            if t.is_empty and not include_empty:
                continue
            cards.append(
                RosterTeamCardSchema(
                    sheet_name=div.sheet_name,
                    team_name=t.team_name,
                    player_count=len(t.players),
                    total_spent=t.total_spent,
                    is_empty=t.is_empty,
                    match_rate=round(t.match_rate, 4),
                ).model_dump(by_alias=True)
            )
    return ORJSONResponse({"contextId": context_id, "teams": cards})


# ── GET /roster/context/{context_id}/team ────────────────────────────────────


@router.get(
    "/context/{context_id}/team",
    response_model=RosterDetailResponseSchema,
    response_class=ORJSONResponse,
    summary="Get full matched roster for one team",
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def get_team_roster(
    request: Request,
    context_id: str,
    sheet_name: str = Query(...),
    team_name: str = Query(...),
) -> RosterDetailResponseSchema:
    store = _get_store(request)
    ctx = store.get(context_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Context non trovato o scaduto. Ricarica il file Excel.",
        )
    team = ctx.get_team(sheet_name, team_name)
    if team is None:
        raise HTTPException(
            status_code=404,
            detail=f"Squadra '{team_name}' non trovata in '{sheet_name}'",
        )

    players: list[RosterPlayerSchema] = []
    for mp in team.players:
        cat = mp.catalog
        players.append(
            RosterPlayerSchema(
                name_raw=mp.parsed.name_raw,
                name_clean=mp.parsed.name_clean,
                cost=mp.parsed.cost,
                status=mp.status.value,
                score=round(mp.score, 4),
                needs_review=mp.needs_review,
                fantacalcio_id=cat.fantacalcio_id if cat else None,
                catalog_name=cat.name if cat else None,
                catalog_team=cat.team if cat else None,
                role_classic=cat.role_classic if cat else None,
                roles_mantra=list(cat.roles_mantra) if cat else [],
            )
        )

    return RosterDetailResponseSchema(
        context_id=context_id,
        sheet_name=sheet_name,
        team_name=team_name,
        total_spent=team.total_spent,
        match_rate=round(team.match_rate, 4),
        players=players,
    )


# ── GET /roster/context/{context_id}/quality ─────────────────────────────────


@router.get(
    "/context/{context_id}/quality",
    response_class=ORJSONResponse,
    summary="Matching quality metrics for a context",
    dependencies=[Depends(rate_limit), Depends(require_role("member"))],
)
async def get_quality(request: Request, context_id: str) -> ORJSONResponse:
    store = _get_store(request)
    ctx = store.get(context_id)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Context non trovato o scaduto. Ricarica il file Excel.",
        )
    return ORJSONResponse(
        {
            "contextId": context_id,
            "quality": _quality_schema(ctx).model_dump(by_alias=True),
            "unmatchedCount": len(ctx.unmatched_players()),
            "provisionalCount": len(ctx.provisional_players()),
        }
    )
