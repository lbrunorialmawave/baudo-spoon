"""Production rollback entry point (WS17 of plan.md).

Questo script è il *canonical* kill-switch per il rollout ML.  Delega
alla CLI ``ml.run_rollout`` (che persiste lo stato e gestisce R2) ma
espone solo i subcommand di rollback per minimizzare il rischio di
``--help`` accidentali in produzione.

Subcommands
-----------
disable
    Forza tutti i flag noti a ``DISABLED`` (kill switch di emergenza).

restore
    Ripristina lo stato da uno snapshot.

list
    Elenca gli snapshot salvati.

save
    Salva uno snapshot dello stato corrente (pre-deploy).

Usage
-----
::

    # Kill switch totale (emergenza):
    python -m ml.scripts.rollback disable \\
        --reason "canary_anomaly in top decile" \\
        --actor lbrunori

    # Snapshot pre-deploy:
    python -m ml.scripts.rollback save \\
        --name pre-shrinkage-2026-08-12 \\
        --actor lbrunori

    # Rollback a snapshot noto:
    python -m ml.scripts.rollback restore \\
        --name pre-shrinkage-2026-08-12 \\
        --reason "anomaly correlata alla nuova promozione" \\
        --actor lbrunori

    # Lista snapshot disponibili:
    python -m ml.scripts.rollback list

Exit codes
----------
0 — successo
1 — errore fatale (R2 sync, eccezione inattesa)
2 — errore di configurazione (env mancanti, parametri invalidi)
3 — promotion gate negato (non applicabile qui, ma riservato per
    coerenza con ``ml.run_rollout``).
"""
from __future__ import annotations

import sys

from ml.run_rollout import main as _run_rollout_main


_COMMANDS: dict[str, str] = {
    "disable": "rollback-all",
    "restore": "restore-snapshot",
    "list": "list-snapshots",
    "save": "save-snapshot",
}

# Opzioni che ``argparse`` di ``ml.run_rollout`` riconosce a livello di
# parent parser (prima del subcommand).  Tutte le altre opzioni sono
# invece definite sul subparser specifico e devono restare DOPO il
# subcommand.  Le manteniamo esplicite per evitare di riordinare
# involontariamente opzioni che ``run_rollout`` non riconosce
# al livello parent.
_PARENT_OPTIONS_WITH_VALUE: frozenset[str] = frozenset({
    "--artifacts-dir",
    "--r2-bucket",
    "--log-level",
})
_PARENT_OPTIONS_FLAG: frozenset[str] = frozenset({
    "--sync-r2",
    "--json-logs",
})


def _split_parent_and_subcommand_args(args: list[str]) -> tuple[list[str], list[str]]:
    """Separa le opzioni del parent parser da quelle del subcommand.

    ``argparse`` rifiuta le opzioni del parent parser se compaiono
    DOPO il subcommand (errore ``unrecognized arguments``), quindi
    quando l'utente scrive ``rollback disable --artifacts-dir /tmp``
    dobbiamo riscrivere l'argv come ``run_rollout --artifacts-dir /tmp
    rollback-all``.
    """
    parent: list[str] = []
    sub_args: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in _PARENT_OPTIONS_WITH_VALUE:
            # Opzione con valore: ``--artifacts-dir PATH``.
            parent.append(tok)
            if i + 1 < len(args):
                parent.append(args[i + 1])
                i += 2
                continue
            i += 1
            continue
        if tok in _PARENT_OPTIONS_FLAG:
            parent.append(tok)
            i += 1
            continue
        if tok.startswith("--") and "=" in tok:
            # Forma ``--artifacts-dir=/tmp/...``: estrai nome e valore.
            name, value = tok.split("=", 1)
            if name in _PARENT_OPTIONS_WITH_VALUE:
                parent.append(name)
                parent.append(value)
                i += 1
                continue
        sub_args.append(tok)
        i += 1
    return parent, sub_args


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__, file=sys.stderr)
        return 0
    sub = sys.argv[1]
    if sub not in _COMMANDS:
        print(
            f"Subcommand '{sub}' non riconosciuto. Valori: {sorted(_COMMANDS)}",
            file=sys.stderr,
        )
        return 2
    parent_opts, sub_opts = _split_parent_and_subcommand_args(sys.argv[2:])
    # Riscrivi l'argv in modo che ``main()`` di ``run_rollout`` veda
    # le opzioni del parent PRIMA del subcommand.
    sys.argv = [sys.argv[0], *parent_opts, _COMMANDS[sub], *sub_opts]
    return _run_rollout_main()


if __name__ == "__main__":
    sys.exit(main())
