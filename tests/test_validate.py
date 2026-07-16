import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validate import validate, EXPECTED_COLUMNS

def make_good_df():
    # minimal valid dataframe with all expected columns
    data = {col: [0.0] for col in EXPECTED_COLUMNS}
    data['Class'] = [0]
    data['Amount'] = [10.0]
    return pd.DataFrame(data)

def test_passes_on_good_data():
    df = make_good_df()
    assert validate(df) == True

def test_catches_missing_column():
    df = make_good_df().drop('Amount', axis=1)
    with pytest.raises(ValueError, match="missing columns"):
        validate(df)

def test_catches_null_values():
    df = make_good_df()
    df.loc[0, 'V1'] = None
    with pytest.raises(ValueError, match="nulls found"):
        validate(df)

def test_catches_bad_class_value():
    df = make_good_df()
    df.loc[0, 'Class'] = 2
    with pytest.raises(ValueError, match="Class column"):
        validate(df)

def test_catches_negative_amount():
    df = make_good_df()
    df.loc[0, 'Amount'] = -5.0
    with pytest.raises(ValueError, match="negative"):
        validate(df)
