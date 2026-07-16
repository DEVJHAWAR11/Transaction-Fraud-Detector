import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# fisher's exact test — better than chi-squared for small fraud counts
# we have ~98 fraud cases in the test set, small enough that fisher's is more accurate

def compute_metrics(df):
    tp = ((df["pred"] == 1) & (df["true_label"] == 1)).sum()
    fp = ((df["pred"] == 1) & (df["true_label"] == 0)).sum()
    fn = ((df["pred"] == 0) & (df["true_label"] == 1)).sum()
    tn = ((df["pred"] == 0) & (df["true_label"] == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # false positive rate
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "fpr": fpr}

# load canary log — has both champion and challenger slices
log = pd.read_csv("data/processed/canary_log.csv")

champ = log[log["model_used"] == "champion"]
chall = log[log["model_used"] == "challenger"]

champ_m = compute_metrics(champ)
chall_m = compute_metrics(chall)

print("=" * 50)
print("A/B ANALYSIS — canary vs champion")
print("=" * 50)
print(f"\nchampion slice ({len(champ)} transactions):")
print(f"  recall:    {champ_m['recall']:.4f}")
print(f"  precision: {champ_m['precision']:.4f}")
print(f"  fpr:       {champ_m['fpr']:.4f}")
print(f"  tp={champ_m['tp']} fp={champ_m['fp']} fn={champ_m['fn']} tn={champ_m['tn']}")

print(f"\nchallenger/canary slice ({len(chall)} transactions):")
print(f"  recall:    {chall_m['recall']:.4f}")
print(f"  precision: {chall_m['precision']:.4f}")
print(f"  fpr:       {chall_m['fpr']:.4f}")
print(f"  tp={chall_m['tp']} fp={chall_m['fp']} fn={chall_m['fn']} tn={chall_m['tn']}")

# fisher's exact test on the 2x2 confusion table for fraud detection
# tests: is the difference in how each model handles fraud cases statistically significant?
# table: [[champion TP, champion FN], [challenger TP, challenger FN]]
# i.e. out of all actual fraud cases, how many did each model catch vs miss?
table = [
    [champ_m["tp"], champ_m["fn"]],
    [chall_m["tp"], chall_m["fn"]]
]

# also test false positive rate difference
# table: [[champion FP, champion TN], [challenger FP, challenger TN]]
table_fpr = [
    [champ_m["fp"], champ_m["tn"]],
    [chall_m["fp"], chall_m["tn"]]
]

odds_recall, p_recall = fisher_exact(table)
odds_fpr, p_fpr = fisher_exact(table_fpr)

print(f"\nstatistical significance (fisher's exact):")
print(f"  recall difference   — p-value: {p_recall:.4f} {'(significant)' if p_recall < 0.05 else '(not significant)'}")
print(f"  fpr difference      — p-value: {p_fpr:.4f} {'(significant)' if p_fpr < 0.05 else '(not significant)'}")
print(f"\n  note: canary slice is only {len(chall)} transactions (~5% of traffic)")
print(f"  small slice = low power — p-values unreliable at this sample size")
print(f"  in production you'd run canary for longer to get more data per slice")
