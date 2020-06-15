import numpy as _np
import pandas as _pd
from sklearn.model_selection._split import _BaseKFold
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


def split_data(df: _pd.DataFrame, ratio: float, purging: bool = True, n_bars: int = 10) -> _Tuple[_pd.DataFrame, _pd.DataFrame]:
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

class PurgedKFold(_BaseKFold):
    """K-Fold with purging and embargo for finantial sequence data
    """
    def __init__(self, n_splits=5, n_overlaps=0, pct_embargo=0.):
        """constructor

        Args:
            n_splits (int, optional): K of the K-fold. Defaults to 5.
            n_overlaps (int, optional): Temporal overlap between adjacent samples in which samples are purged. Defaults to 0.
            pct_embargo ([type], optional): Percent of embargo. Defaults to 0..
        """
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.n_overlaps = n_overlaps
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        train_ratio = (self.n_splits - 1) / self.n_splits
        indices = _np.arange(X.shape[0])
        n_embargo = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1]+1) for i in _np.array_split(indices, self.n_splits)]
        for i, j in test_starts:
            if self.n_overlaps > 0:
                train_f_purge = round(self.n_overlaps * train_ratio)
                train_f_idx1 = i - train_f_purge
                test_idx0 = i + (self.n_overlaps - train_f_purge)
            else:
                train_f_idx1 = test_idx0 = i

            if n_embargo > 0:
                purge_range = self.n_overlaps + n_embargo
                train_l_purge = round(purge_range * train_ratio)
                test_idx1 = j - (purge_range - train_l_purge)
                train_l_idx0 = j + train_l_purge
            else:
                test_idx1 = train_l_idx0 = j

            train_f_indices = _pd.Series(X.iloc[0:train_f_idx1].index)
            test_indices = _pd.Series(X.iloc[test_idx0:test_idx1].index)
            train_l_indices = _pd.Series(X.iloc[train_l_idx0:].index)
            train_indices = _pd.concat([train_f_indices, train_l_indices])
            yield train_indices, test_indices

if __name__ == '__main__':
    df = _pd.DataFrame({'Date': ['12/23/1991', '12/24/1991', '12/25/1991'],
                        'Open': _np.arange(3.0), 'Close': _np.arange(10.0, 13)})
    expected = _pd.DataFrame({
        'Date': ['12/25/1991'],
        'Open2': 0.0,
        'Close2': 10.0,
        'Open1': 1.0,
        'Close1': 11.0,
        'Open0': 2.0,
        'Close0': 12.0
    })
    assert (expected.equals(make_nbars(df, 2, cols=[
            'Open', 'Close'], datetime_col='Date')))
