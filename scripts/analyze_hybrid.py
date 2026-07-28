"""Analyze hybrid predictions data from live API."""
import json
import urllib.request

url = "https://baudo-spoon.onrender.com/api/v1/predictions/hybrid?page=1&size=2000"
with urllib.request.urlopen(url, timeout=30) as resp:
    d = json.loads(resp.read().decode("utf-8"))

items = d["items"]
total = len(items)

no_fp = [p for p in items if p.get("FP_Corr") is None]
no_pred = [p for p in items if p.get("predictedFantavoto") is None]
no_ml = [p for p in items if not p.get("hasMlData")]
boosted = [p for p in items if "ML_Boosted" in (p.get("hybridLabels") or [])]
contra = [p for p in items if "Contradiction" in (p.get("hybridLabels") or [])]
confirmed = [p for p in items if "ML_Confirmed" in (p.get("hybridLabels") or [])]
risky = [p for p in items if "ML_Risky" in (p.get("hybridLabels") or [])]
sleeper = [p for p in items if "Sleeper" in (p.get("hybridLabels") or [])]
best_val = [p for p in items if "Best_Value" in (p.get("hybridLabels") or [])]
min_risk = [p for p in items if "Minutes_Risk" in (p.get("hybridLabels") or [])]

print(f"Total players: {total}")
print(f"  No FP_Corr:         {len(no_fp):>4} ({len(no_fp)/total*100:.1f}%)")
print(f"  No predicted:       {len(no_pred):>4}")
print(f"  No hasMlData:       {len(no_ml):>4}")
print(f"  ML_Boosted:         {len(boosted):>4}")
print(f"  Contradiction:      {len(contra):>4}")
print(f"  ML_Confirmed:       {len(confirmed):>4}")
print(f"  ML_Risky:           {len(risky):>4}")
print(f"  Sleeper:            {len(sleeper):>4}")
print(f"  Best_Value:         {len(best_val):>4}")
print(f"  Minutes_Risk:       {len(min_risk):>4}")

if no_fp:
    print("\n=== No FP_Corr sample ===")
    for p in no_fp[:5]:
        print(f'  {p["playerName"]:25s} team={p["team"]:15s} ruolo={p["ruoloPrimario"] or "?":4s} '
              f'hasMl={p["hasMlData"]} fpIbrido={p["fpIbrido"]} pred={p["predictedFantavoto"]} '
              f'P1={p.get("P1")} P2={p.get("P2")} P3={p.get("P3")} P4={p.get("P4")} '
              f'CP={p.get("CP")} FP={p.get("FP")}')

if no_fp:
    print("\n=== FP_Corr null: check if P1/P2/P3/P4 are all null ===")
    all_p_null = [p for p in no_fp if p.get("P1") is None and p.get("P2") is None and p.get("P3") is None]
    some_p = [p for p in no_fp if p.get("P1") is not None or p.get("P2") is not None]
    print(f"  All 4 pillars null: {len(all_p_null)}")
    print(f"  Some pillar present: {len(some_p)}")
    if some_p:
        print("  Sample with some pillars:")
        for p in some_p[:3]:
            print(f'    {p["playerName"]:25s} P1={p.get("P1")} P2={p.get("P2")} P3={p.get("P3")} P4={p.get("P4")} CP={p.get("CP")} FP={p.get("FP")}')

print("\n=== ML_Boosted sample (first 10) ===")
for p in boosted[:10]:
    gap = p.get("fpGap")
    print(f'  {p["playerName"]:25s} fpIbrido={p["fpIbrido"]:>6.1f} pred={p["predictedFantavoto"]:>5} '
          f'fpCorr={str(p.get("FP_Corr")):>6} gap={str(gap):>6} '
          f'mlBoost={p.get("mlBoost")} conf={p.get("confidenceScore")}')

print("\n=== Labels overlap analysis ===")
multi_label = [p for p in items if len(p.get("hybridLabels") or [] ) > 1]
print(f"Players with 2+ labels: {len(multi_label)}")
boosted_and_contra = [p for p in items if "ML_Boosted" in (p.get("hybridLabels") or []) and "Contradiction" in (p.get("hybridLabels") or [])]
print(f"ML_Boosted + Contradiction: {len(boosted_and_contra)}")
if boosted_and_contra:
    print("  Sample:")
    for p in boosted_and_contra[:5]:
        print(f'    {p["playerName"]:25s} fpIbrido={p["fpIbrido"]} fpCorr={p.get("FP_Corr")} pred={p["predictedFantavoto"]} gap={p.get("fpGap")}')

print("\n=== Predicted distribution ===")
preds = [p["predictedFantavoto"] for p in items if p["predictedFantavoto"] is not None]
if preds:
    preds.sort()
    print(f"  Min: {preds[0]:.2f}, Max: {preds[-1]:.2f}, Med: {preds[len(preds)//2]:.2f}")
    print(f"  Count with pred: {len(preds)}/{total}")
