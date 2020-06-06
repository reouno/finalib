import numpy as np
import pandas as pd
from typing import List, Text, Union

def make_nbars(df: pd.DataFrame, n_bars: int, cols: List[Text]=['Close'], datetime_col: Union[Text, None]='Date') -> pd.DataFrame:
    '''Make n bars dataframe.
    The row size of `df` must be greater than or equal to `n_bars`, or raise ValueError.
    '''
    if df.shape[0] < n_bars + 1:
        raise ValueError(f'row size of the df (={df.shape[0]}) must be greater than or equal to n_bars + 1 (={n_bars + 1})')
    df = df.rename(columns={ col:f'{col}{n_bars}' for col in cols })

    for i in reversed(range(n_bars)):
        inc = n_bars - i
        for col in cols:
            df[f'{col}{i}'] = df[f'{col}{n_bars}'][inc:].append(pd.Series([np.nan]*inc)).reset_index(drop=True)

    # correct bar date (or datetime)
    if datetime_col is not None:
        df[datetime_col] = df[datetime_col][n_bars:].append(pd.Series(['-']*n_bars)).reset_index(drop=True)

    # delete last n rows as they have nan values
    df = df[:-n_bars]

    return df