import pandas as pd
import numpy as np
import os
import logging
from ..features import FeatureEncoder as BaseFeatureEncoder
from datetime import datetime, date


class FeatureEncoder(BaseFeatureEncoder):

    # row filtering before column preprocessing, this function is always called
    def preprocess(self, ddf, fill_na=True):
        return ddf

    def logar(self, df, col_name):
        def _to_log(value):
            return np.log(value + 2)
        return df[col_name].map(_to_log).astype(float)
    
    def encode_categories(self, df, col_name):
        return df[col_name].apply(lambda x: f"{col_name}_{x}")

    def bucketize(self, df, col_name):
        return pd.cut(df[col_name], 5)
