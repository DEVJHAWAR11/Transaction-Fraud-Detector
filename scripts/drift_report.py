import pandas as pd
import numpy as np
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset
import os

# reference = training data (what the model was trained on)
# current = simulated "new batch" of incoming transactions
#
# how the drift is simulated:
# we take a slice of the test set and add controlled noise to Amount and V1/V17
# Amount gets a +50 shift (simulates a period with higher transaction values)
# V1 and V17 get gaussian noise added (simulates feature distribution shift)
# this is a realistic simulation — in production, drift happens when transaction
# patterns change (new merchants, seasonal spending, new fraud patterns)
# we're honest about this: it's synthetic, but the shifts are meaningful

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# use first 5000 rows of train as reference (manageable size for the report)
reference = X_train.sample(5000, random_state=42).reset_index(drop=True)

# current batch: 2000 rows from test set with synthetic drift applied
current = X_test.sample(2000, random_state=99).reset_index(drop=True)

np.random.seed(42)
# shift Amount upward — simulates a period with higher average transaction values
current["Amount"] = current["Amount"] + 50 + np.random.normal(0, 10, len(current))

# add noise to V1 and V17 — the two features most correlated with fraud
# simulates a shift in the fraud pattern fingerprint
current["V1"] = current["V1"] + np.random.normal(0.5, 0.3, len(current))
current["V17"] = current["V17"] + np.random.normal(0.3, 0.2, len(current))

print(f"reference batch: {len(reference)} rows")
print(f"current batch:   {len(current)} rows")
print(f"\nAmount mean — reference: {reference['Amount'].mean():.3f}, current: {current['Amount'].mean():.3f}")
print(f"V1 mean     — reference: {reference['V1'].mean():.3f}, current: {current['V1'].mean():.3f}")
print(f"V17 mean    — reference: {reference['V17'].mean():.3f}, current: {current['V17'].mean():.3f}")

# only use a subset of columns for the report — all 30 would make it huge
cols_to_check = ["Amount", "Time", "V1", "V4", "V10", "V12", "V14", "V17"]
ref_ds = Dataset.from_pandas(reference[cols_to_check])
cur_ds = Dataset.from_pandas(current[cols_to_check])

report = Report([DataDriftPreset()])
result = report.run(reference_data=ref_ds, current_data=cur_ds)

os.makedirs("reports", exist_ok=True)
result.save_html("reports/drift_report.html")
print("\ndrift report saved to reports/drift_report.html")
