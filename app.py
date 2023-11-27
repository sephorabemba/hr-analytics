import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import train_filepath, valid_filepath, test_filepath, model_filepath
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components
import plotly.express as px
import altair as alt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.title("🔍 Employee turnover analysis 👩‍💼")

st.subheader("What is this app?", divider=True)

summary = f""" ### Business case 
Clover Electrical Inc. is a tech company specialized in sustainable electronics. The 
company has been suffering from an increasing turnover rate for over a year now. The past period, it has stagnated 
around a whooping 16%.
 
Leadership has tried to set up actions but they were unsuccessful. They seemingly didn't address the core issues.

To make more data-driven decisions, they've sent an employee survey to the company to better understand the situation
and target the right problems this time on.

The leadership team has asked the Data team to analyze the results and build a solution to predict employee turnover,
uncover its main factors and make recommendations on how to reduce employee turnover.

### Goals
* Predict turnover with a 90% average precision.
* Uncover impactful insights and recommendations to increase employee retention.

### Business understanding
Retaining talents in a company is important to build steady businesses. Many companies struggle with preventing turnover.
The first reasons are lack of understanding the factors, lack of timeliness of actions, 
actions are not addressing the core problems.

High turnover must be addressed as it is:
- Disruptive for the teams, impacts group morale negatively
- Costly for the company: it will cost your business, on average, \$25,000 to $100,000 per employee.
- Increase product failure by 0.74% to 0.79%: which amounts to millions of dollars.

Once again, data analysis, exploration and modeling techniques are key to crack this problem.

### The approach
The data team has thoroughly analyzed the survey data and has come up with a HR analytics software powered with AI to
solve this business problem (*Use 55 template*).

### Techniques used:
- Exploratory Data Analysis: Python, scikit learn
- Machine Learning modeling: Gradient Boosting algorithm
- Web app development & deployment: Streamlit, Streamlit Cloud

This use case is based on the "Hr Analytics Job Prediction" dataset available on [Kaggle](
https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)."""

st.markdown(summary)


@st.cache_resource
def load_model():
    return joblib.load(filename=model_filepath)


@st.cache_data
def predict(_pipeline, X):
    return pipeline.predict(X)


@st.cache_data
def load_datasets():
    df_train = pd.read_csv(train_filepath)
    df_valid = pd.read_csv(valid_filepath)
    df_test = pd.read_csv(test_filepath)

    return df_train, df_valid, df_test


@st.cache_data
def get_feats_targets():
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_valid = df_valid.iloc[:, :-1]
    y_valid = df_valid.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


@st.cache_data
def compute_turnover(y_labels):
    """

    :return:
    """
    turnover_nb = y_labels.sum()
    turnover_prc = turnover_nb / len(y_labels) * 100
    return turnover_prc, turnover_nb


@st.cache_data
def compare_turnover(turnover_actual, turnover_pred, turnover_count_actual, turnover_count_pred, nb_employees):
    """

    :return:
    """
    delta = np.abs(turnover_pred - turnover_actual)
    return (f"""
    Total number of employees Q3: {nb_employees}
    Actual turnover: {turnover_actual:.2f}% ({turnover_count_actual}/{nb_employees} people)
    Predicted turnover: {turnover_pred:.2f}% ({turnover_count_pred}/{nb_employees} people)
    Delta: {delta:.2f} pt
    """)


# ======= MAIN =======

# load and prepare data
df_train, df_valid, df_test = load_datasets()
X_train, y_train, X_valid, y_valid, X_test, y_test = get_feats_targets()
pipeline = load_model()
st.dataframe(df_test)

# predict
y_train_pred = predict(_pipeline=pipeline, X=X_train)
y_test_pred = predict(_pipeline=pipeline, X=X_test)
turnover_train_actual, nb_tra = compute_turnover(y_labels=y_train)
turnover_train_pred, nb_trp = compute_turnover(y_labels=y_train_pred)
turnover_test_actual, nb_tea = compute_turnover(y_labels=y_test)
turnover_test_pred, nb_tep = compute_turnover(y_labels=y_test_pred)

# ========== COMPARISON PREDICTED VS ACTUAL ==========
st.subheader("Turnover rates Actual vs Predicted", divider=True)

# column layout
col_main, col_zoom = st.columns([0.6, 0.4])

# TODO: simple dual bar chart over % with comparison with increase animation
with col_main:
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    df_turnover = pd.DataFrame(
        data=[[round(turnover_test_actual, 2), "Actual"], [round(turnover_test_pred, 2), "Algo prediction"]],
        columns=["Turnover (%)", "Source"],
    )
    chart = (
        alt.Chart(df_turnover)
        .mark_bar(size=50)
        .encode(
            alt.X("Source:N", axis=alt.Axis(labelAngle=0)),
            alt.Y("Turnover (%):Q"),
            alt.Color("Source:N"),
            alt.Tooltip(["Turnover (%)", "Source"]),
        ).properties(
            width=350,
            height=400,
            title="Turnover rates - Q3"
        ).interactive()
    )
    st.altair_chart(chart)

# =================== ZOOM ON PREDICTIONS ===================
with col_zoom:
    col_proportion = "Proportion (%)"
    col_source = "Source"
    #TODO: stacked bar plot with each proportions
    df_zoom = pd.DataFrame(
        data= [[ 80, "Well Predicted","Segment"],[ 8,"Safe marked as Leavers","Segment"], [12, "Missed Leavers","Segment"]],
        columns=[col_proportion, col_source, "Group"]
    ).sort_values(by=col_proportion, ascending=False)

    # stacked by Source
    domain = ["Well Predicted", "Safe marked as Leavers", "Missed Leavers"]
    range_colors = ["#32a838", "#e8911e", "#e8251e"]
    chart = (
        alt.Chart(df_zoom)
        .mark_bar(size=100)
        .encode(
            #x=alt.X("Group:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{col_proportion}:Q"),
            color=alt.Color(f"{col_source}:N", scale=alt.Scale(domain=domain, range=range_colors)),
            tooltip=alt.Tooltip([col_proportion, col_source]),
        ).properties(
            width=300,
            height=400,
            title="Zoom on predictions"
        ).interactive()
    )
    st.altair_chart(chart)







st.text(f"Latest turnover data - Period Q2")
st.text(f"{turnover_train_actual:.2f}")

st.text(f"Employees at risk actual - Period Q3")
st.text(f"{turnover_test_actual:.2f}")

st.text(f"Employees at risk predicted - Period Q3")
st.text(f"{turnover_test_pred:.2f}")

# show comparison
comparison = compare_turnover(
    turnover_actual=turnover_test_actual,
    turnover_pred=turnover_test_pred,
    turnover_count_actual=nb_tea,
    turnover_count_pred=nb_tep,
    nb_employees=len(y_test)
)
st.text(comparison)

# ========== TRENDS ==========
# add graph interactive with prediction point vs actual and show trend
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

turnover_q1_actual = 15  # dummy
df_trend_actual = pd.Series([15, 14.6, turnover_test_actual], name="Actual trend")
df_trend_predict = pd.Series([15, 14.6, turnover_test_pred], name="Predicted trend")
df_trend = pd.concat([df_trend_actual, df_trend_predict], axis=1)
df_trend

fig = plt.figure()
plt.plot(df_trend)
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)

fig = px.line(df_trend, title='Turnover trend')
fig.update_layout(
    autosize=False,
    width=800,
    height=800,
)
st.plotly_chart(fig, use_container_width=True)
# st.line_chart(df_trend)
