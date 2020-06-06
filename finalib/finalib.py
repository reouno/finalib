import numpy as _np
import pandas as _pd
from typing import List as _List
from typing import Text as _Text
from typing import Union as _Union


def make_nbars(df: _pd.DataFrame, n_bars: int, cols: _List[_Text] = ['Close'], datetime_col: _Union[_Text, None] = 'Date') -> _pd.DataFrame:
    '''Make n bars dataframe.
    The row size of `df` must be greater than or equal to `n_bars`, or raise ValueError.
    '''
    if df.shape[0] < n_bars + 1:
        raise ValueError(
            f'row size of the df (={df.shape[0]}) must be greater than or equal to n_bars + 1 (={n_bars + 1})')
    df = df.rename(columns={col: f'{col}{n_bars}' for col in cols})

    for i in reversed(range(n_bars)):
        inc = n_bars - i
        for col in cols:
            df[f'{col}{i}'] = df[f'{col}{n_bars}'][inc:].append(
                _pd.Series([_np.nan]*inc)).reset_index(drop=True)

    # correct bar date (or datetime)
    if datetime_col is not None:
        df[datetime_col] = df[datetime_col][n_bars:].append(
            _pd.Series(['-']*n_bars)).reset_index(drop=True)

    # delete last n rows as they have nan values
    df = df[:-n_bars]

    return df

if __name__ == '__main__':
    df = _pd.DataFrame({'Date':['12/23/1991','12/24/1991','12/25/1991'], 'Open':_np.arange(3.0), 'Close':_np.arange(10.0,13)})
    expected = _pd.DataFrame({
        'Date':['12/25/1991'],
        'Open2': 0.0,
        'Close2': 10.0,
        'Open1': 1.0,
        'Close1': 11.0,
        'Open0': 2.0,
        'Close0': 12.0
    })
    assert (expected.equals(make_nbars(df, 2, cols=['Open','Close'], datetime_col='Date')))