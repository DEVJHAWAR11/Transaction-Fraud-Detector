import pandas as pd

EXPECTED_COLUMNS = (
    ['Time'] +
    [f'V{i}' for i in range(1, 29)] +
    ['Amount', 'Class']
)

def validate(df):
    # check all columns are present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    # no nulls allowed
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"nulls found in columns: {bad.to_dict()}")

    # Class must be 0 or 1 only
    bad_class = df[~df['Class'].isin([0, 1])]
    if not bad_class.empty:
        raise ValueError(f"Class column has values other than 0/1: {df['Class'].unique().tolist()}")

    # amount can't be negative
    if (df['Amount'] < 0).any():
        raise ValueError("Amount column has negative values")

    return True
