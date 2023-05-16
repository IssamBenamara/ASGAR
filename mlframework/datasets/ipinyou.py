import pandas as pd
import numpy as np
import os
from ..features import FeatureEncoder as BaseFeatureEncoder
from datetime import datetime, date


class FeatureEncoder(BaseFeatureEncoder):
    def to_log(self, df, col_name):
        def _to_log(value):
            return np.log(value)
        return df[col_name].map(_to_log).astype(float)
    
    def bucketize_payprice(self, df, col_name):
        def _bucketize(value):
            if pd.isnull(value):
                return 0
            else:
                value = float(value)
                if value <= 54.2:
                    return 0
                elif 54.2 < value <= 107.4:
                    return 1
                elif 107.4 < value <= 160.6:
                    return 2
                elif 160.6 < value <= 213.8:
                    return 3
                elif 213.8 < value:
                    return 4

        return df[col_name].apply(_bucketize)

    def bucketize(self, df, col_name):
        
        return pd.cut(df[col_name], 5)