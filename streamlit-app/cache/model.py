import joblib
import pandas as pd
import streamlit as st

from utils.utils import (
    model_filepath,
    Factor,
)


@st.cache_resource
def load_model():
    return joblib.load(filename=model_filepath)


@st.cache_data
def predict(_pipeline, X):
    return _pipeline.predict(X)


@st.cache_data
def get_feat_importance(_pipeline):
    features = _pipeline.feature_names_in_
    importance = _pipeline[-1].feature_importances_.round(2) * 100
    df_importance = pd.DataFrame({"Importance (%)": importance}, index=features)
    df_importance = df_importance.reset_index(). \
        rename({"index": "Feature"}, axis=1). \
        sort_values(by="Importance (%)", ascending=False).reset_index(drop=True).reset_index().rename({"index": "rank"},
                                                                                                      axis=1)
    df_importance["Factor"] = df_importance["Feature"].apply(lambda x: Factor.format_factor(factor=x))
    return df_importance
