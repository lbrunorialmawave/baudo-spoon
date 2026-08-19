"""Tests for process-local RosterContextStore."""

from __future__ import annotations

import time

from ml.roster_import import (
    CatalogPlayer,
    build_roster_context,
    parse_bytes,
)
from ml.roster_import.store import RosterContextStore, reset_default_store
import openpyxl
import io


def _tiny_workbook() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Divisione A"
    ws.append(["Alpha FC", "costo", None])
    ws.append(["De Gea", 10, None])
    ws.append(["Barella", 20, None])
    ws.append(["totale", 30, None])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _catalog():
    return [
        CatalogPlayer(1, "De Gea", "Fiorentina", "P", ("Por",)),
        CatalogPlayer(2, "Barella", "Inter", "C", ("C",)),
    ]


def test_put_get_update_delete():
    store = RosterContextStore(ttl_seconds=120)
    wb = parse_bytes(_tiny_workbook(), source_filename="t.xlsx")
    ctx = build_roster_context(wb, _catalog())
    cid = store.put(ctx)
    assert store.get(cid) is ctx or store.get(cid).context_id == cid

    ctx2 = ctx.with_user_team("Divisione A", "Alpha FC")
    store.update(ctx2)
    got = store.get(cid)
    assert got is not None
    assert got.user_team_key == "Divisione A::Alpha FC"

    assert store.delete(cid) is True
    assert store.get(cid) is None
    assert store.delete(cid) is False


def test_ttl_expiry():
    store = RosterContextStore(ttl_seconds=1)
    wb = parse_bytes(_tiny_workbook())
    ctx = build_roster_context(wb, _catalog())
    cid = store.put(ctx)
    assert store.get(cid) is not None
    # force expire
    store._data[cid].expires_at = time.monotonic() - 1
    assert store.get(cid) is None


def test_size_purges_expired():
    store = RosterContextStore(ttl_seconds=60)
    wb = parse_bytes(_tiny_workbook())
    ctx = build_roster_context(wb, _catalog())
    store.put(ctx)
    assert store.size() == 1
    # inject expired
    store._data["dead"] = type(store._data[ctx.context_id])(
        context=ctx, expires_at=time.monotonic() - 10
    )
    assert store.size() == 1  # purge on size()


def test_reset_default_store():
    reset_default_store()
    from ml.roster_import.store import get_default_store

    s1 = get_default_store()
    s2 = get_default_store()
    assert s1 is s2
    reset_default_store()
