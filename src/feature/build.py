from typing import List
import pandas
from pathlib import Path
from xfeat import SelectCategorical, ConcatCombination, LabelEncoder, Pipeline, SelectNumerical


def read_df(root_dir:str, path:str, categorical_cols:List[str], exclude_cols:List[str]=["imp_time"]):
    df = pandas.read_csv(Path(root_dir, path))
    df = df.astype({c:str for c in categorical_cols})
    encoder = Pipeline([
        SelectCategorical(exclude_cols=exclude_cols),
        LabelEncoder(output_suffix=""),
    ])
    df_encoded = pandas.concat([SelectNumerical().fit_transform(df), encoder.fit_transform(df)], axis=1)    
    return df_encoded