
import pathlib

import pandas as pd

__all__ = ['LoadIceCream']


def LoadIceCream():
    """The amount of ice cream I've eaten in the last three years.

    Example:

    >>> from arbayes import dataset

    >>> dataset.LoadIceCream().head()
       time     y
    0  2017/01  10
    1  2017/02  20
    2  2017/03  25
    3  2017/04  45
    4  2017/05  75

    """
    path = pathlib.Path(__file__).parent.joinpath('ice_cream.csv')
    return pd.read_csv(path)
