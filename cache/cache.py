import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.utils import (
    train_filepath,
    valid_filepath,
    test_filepath,
    model_filepath,
    dept_map,
    segments,
    Perf,
    Salary,
    Tenure,
    Factor,
    recos_filepath,
)



@st.cache_data
def load_datasets():
    df_train = pd.read_csv(train_filepath)
    df_valid = pd.read_csv(valid_filepath)
    df_test = pd.read_csv(test_filepath)

    # add formatted department
    df_train["department_fmt"] = df_train["department"].replace(dept_map)
    df_valid["department_fmt"] = df_valid["department"].replace(dept_map)
    df_test["department_fmt"] = df_test["department"].replace(dept_map)

    # add tenure groups
    df_train["tenure_fmt"] = df_train["tenure"].apply(lambda tenure: Tenure.format_tenure(tenure))
    df_valid["tenure_fmt"] = df_valid["tenure"].apply(lambda tenure: Tenure.format_tenure(tenure))
    df_test["tenure_fmt"] = df_test["tenure"].apply(lambda tenure: Tenure.format_tenure(tenure))

    # add perf groups
    df_train["perf_fmt"] = df_train["last_evaluation"].apply(lambda perf: Perf.format_perf(perf=perf))
    df_valid["perf_fmt"] = df_valid["last_evaluation"].apply(lambda perf: Perf.format_perf(perf=perf))
    df_test["perf_fmt"] = df_test["last_evaluation"].apply(lambda perf: Perf.format_perf(perf=perf))

    # add formatted salary
    df_train["salary_fmt"] = df_valid["salary"].apply(lambda salary: Salary.format_salary(salary))
    df_valid["salary_fmt"] = df_valid["salary"].apply(lambda salary: Salary.format_salary(salary))
    df_test["salary_fmt"] = df_test["salary"].apply(lambda salary: Salary.format_salary(salary))

    # rearrange cols
    cols = df_test.columns.tolist()
    cols.pop(cols.index("left"))
    df_train = df_train[cols + ["left"]]
    df_valid = df_valid[cols + ["left"]]
    df_test = df_test[cols + ["left"]]

    return df_train, df_valid, df_test


@st.cache_data
def get_feats_targets(df_train, df_valid, df_test):
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_valid = df_valid.iloc[:, :-1]
    y_valid = df_valid.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


@st.cache_data
def load_full_data():
    df_train, df_valid, df_test = load_datasets()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_feats_targets(df_train, df_valid, df_test)

    dict_data = {
        "df_train": df_train,
        "df_valid": df_valid,
        "df_test": df_test,
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
    }
    return dict_data

@st.cache_resource
def load_model():
    return joblib.load(filename=model_filepath)

@st.cache_data
def predict(_pipeline, X):
    return _pipeline.predict(X)


@st.cache_data
def compute_turnover(y_labels):
    """

    :return:
    """
    turnover_nb = y_labels.sum()
    turnover_prc = turnover_nb / len(y_labels) * 100
    return turnover_prc, turnover_nb



