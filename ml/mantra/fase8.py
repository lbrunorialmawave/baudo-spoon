"""Fase 8 — Classifications (filters that do not alter scores).

Categories
----------
A.  Top per ruolo (primario) — top N by FP_Mantra in each role
A2. Per ruolo Mantra (multi-eleggibilità) — same player in all roles
B.  *(free — CERTEZZA promoted to Fase 7)*
C.  Low Cost: Prezzo_Massimo <= soglia AND VR > 110
D.  Low Cost Titolari: like C + Pr >= 0.65
E.  Scommesse Multi-ruolo: among SCOMMESSA, priority to Num_Ruoli >= 2
F.  Watchlist Giovani: Eta <= 23 AND Trend > 0
G.  Rischio Contestuale: flag "cambio squadra" / "allenatore instabile"
H.  Consigliati per la giornata: starter with high FP (needs matchday data)
I.  Indisponibili: injured / suspended / doubtful (needs matchday data)
"""

from __future__ import annotations

import pandas as pd


def top_per_ruolo(
    df: pd.DataFrame,
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    """A. Top per ruolo primario — top N per FP_Mantra in ogni ruolo."""
    result: dict[str, pd.DataFrame] = {}
    for ruolo in df["ruolo_primario"].unique():
        subset = df[df["ruolo_primario"] == ruolo].nlargest(top_n, "fp_mantra")
        result[ruolo] = subset
    return result


def multi_eleggibilita(
    df: pd.DataFrame,
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    """A2. Per ruolo Mantra (multi-eleggibilità).

    Each player appears in every role they can cover, not just the primary one.
    Scores shown are those computed on the player's primary role.
    """
    result: dict[str, list] = {r: [] for r in df["ruolo_primario"].unique()}
    # Also include roles only present in ruoli_mantra
    all_roles = set(df["ruolo_primario"].unique())
    for roles_list in df["ruoli_mantra"].dropna():
        all_roles.update(roles_list)

    result = {r: [] for r in all_roles}

    for _, row in df.iterrows():
        roles = row.get("ruoli_mantra") or [row["ruolo_primario"]]
        for r in roles:
            result.setdefault(r, []).append(row)

    # Sort and trim each role list
    sorted_result: dict[str, pd.DataFrame] = {}
    for ruolo, players in result.items():
        if not players:
            continue
        subset = pd.DataFrame(players).nlargest(top_n, "fp_mantra")
        sorted_result[ruolo] = subset
    return sorted_result


def low_cost(
    df: pd.DataFrame,
    soglia_prezzo: float = 15.0,
    vr_soglia: float = 110.0,
    require_titolare: bool = False,
    pr_soglia: float = 0.65,
) -> pd.DataFrame:
    """C. Low Cost — Prezzo_Massimo <= soglia AND VR > 110.
    D. Low Cost Titolari — same + Pr >= 0.65 (use require_titolare=True).
    """
    mask = (df["prezzo_massimo"] <= soglia_prezzo) & (df["vr"] > vr_soglia)
    if require_titolare:
        pr = df.get("Pr", pd.Series(0, index=df.index)).fillna(0)
        mask = mask & (pr >= pr_soglia)
    return df[mask].sort_values("vr", ascending=False)


def scommesse_multi_ruolo(
    df: pd.DataFrame,
    fase7_label: pd.Series,
) -> pd.DataFrame:
    """E. Scommesse Multi-ruolo — among SCOMMESSA, priority to Num_Ruoli >= 2."""
    scommesse = df[fase7_label == "SCOMMESSA"].copy()
    n_ruoli = scommesse.get("Num_Ruoli", pd.Series(1, index=scommesse.index)).fillna(1)
    scommesse["_priority"] = (n_ruoli >= 2).astype(int)
    return scommesse.sort_values(["_priority", "vr"], ascending=[False, False]).drop(
        columns="_priority"
    )


def watchlist_giovani(
    df: pd.DataFrame,
    eta_max: int = 23,
) -> pd.DataFrame:
    """F. Watchlist Giovani — Eta <= 23, sorted by VR descending."""
    eta = df.get("Eta", pd.Series(99, index=df.index)).fillna(99)
    mask = eta <= eta_max
    return df[mask].sort_values("vr", ascending=False)


def rischio_contestuale(
    df: pd.DataFrame,
) -> pd.Series:
    """G. Rischio Contestuale — textual flags, does not alter scores."""
    cambio = df.get("Cambio_Squadra", pd.Series(False, index=df.index)).fillna(False)

    flags_list: list[str | None] = []
    for i in range(len(df)):
        flags: list[str] = []
        if cambio.iloc[i] is True or cambio.iloc[i] == "Si":
            flags.append("Cambio Squadra")
        if len(flags) == 0:
            flags_list.append(None)
        else:
            flags_list.append("; ".join(flags))
    return pd.Series(flags_list, index=df.index)


def consigliati_giornata(
    df: pd.DataFrame,
    probability_col: str = "matchday_probability",
    soglia_prob: float = 70.0,
) -> pd.DataFrame:
    """H. Consigliati per la giornata — probability >= soglia, sorted by FP_Mantra.

    Requires matchday data (from probabili formazioni scraper) to be merged.
    """
    prob = df.get(probability_col, pd.Series(0, index=df.index)).fillna(0)
    mask = prob >= soglia_prob
    return df[mask].sort_values("fp_mantra", ascending=False)


def indisponibili(
    df: pd.DataFrame,
    status_col: str = "matchday_status",
) -> pd.DataFrame:
    """I. Indisponibili — injured / suspended / doubtful.

    Requires matchday data merged.
    """
    status = df.get(status_col, pd.Series("", index=df.index)).fillna("")
    mask = status.isin(["injured", "suspended", "doubtful"])
    return df[mask]
