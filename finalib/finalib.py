import numpy as _np
import pandas as _pd
from typing import List as _List
from typing import Text as _Text
from typing import Tuple as _Tuple
from typing import Union as _Union


def make_nbars(df: _pd.DataFrame, n_bars: int, cols: _List[_Text] = ['Close'], datetime_col: _Union[_Text, None] = 'Date') -> _pd.DataFrame:
    """Make n bars dataframe.
    The row size of `df` must be greater than or equal to `n_bars`, or raise ValueError.
    """
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


def split_data(df: _pd.DataFrame, ratio: float, purging: bool=True, n_bars: int=10) -> _Tuple[_pd.DataFrame, _pd.DataFrame]:
    """split data into two

    Args:
        df (DataFrame): Time-seriese data that is to splitted.
        ratio (float): Ratio of the first sub data of the splitted data.
        purging (bool, optional): Purge samples with overlapping periods. Defaults to True.
        n_bars (int, optional): Period of one sample. Defaults to 10.

    Returns:
        Tuple[_pd.DataFrame, _pd.DataFrame]: Pair of the splitted data.
    """
    split_idx = int(df.shape[0] * ratio)
    df1 = df[:split_idx]
    df2 = df[split_idx:]
    if purging:
        purge_idx = round((n_bars-1) * ratio)
        df1 = df1[:-purge_idx]
        df2 = df2[(n_bars - 1 - purge_idx):]

    return df1, df2


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