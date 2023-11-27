import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import train_filepath, valid_filepath, test_filepath, model_filepath, dept_map, segments
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components
import plotly.express as px
import altair as alt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

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

    df_train["department_fmt"] = df_train["department"].replace(dept_map)
    df_valid["department_fmt"] = df_valid["department"].replace(dept_map)
    df_test["department_fmt"] = df_test["department"].replace(dept_map)

    # rearrange cols
    cols = df_test.columns.tolist()
    cols.pop(cols.index("left"))
    df_train = df_train[cols + ["left"]]
    df_valid = df_valid[cols + ["left"]]
    df_test = df_test[cols + ["left"]]

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
txt_period = "**Period:** Q3"
txt_workforce = f"**Workforce size:** {len(y_test)}"

col_period, col_workforce = st.columns([0.15, 0.85])
with col_period:
    st.markdown(txt_period)
with col_workforce:
    st.markdown(txt_workforce)

# column layout
col_main, col_zoom = st.columns([0.5, 0.5])

# TODO: simple dual bar chart over % with comparison with increase animation
with col_main:
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    df_turnover = pd.DataFrame(
        data=[[round(turnover_test_actual, 2), "Actual"], [round(turnover_test_pred, 2), "Algo predictions"]],
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
            title="Turnover rates"
        ).interactive()
    )
    st.altair_chart(chart)

# =================== ZOOM ON PREDICTIONS ===================
with col_zoom:
    col_proportion = "Proportion (%)"
    col_source = "Source"
    col_group = "Predictions Breakdown"
    # TODO: stacked bar plot with each proportions
    df_zoom = pd.DataFrame(
        data=[[80, "Well Predicted", "Segment"], [8, "Safe marked as Leavers", "Segment"],
              [12, "Missed Leavers", "Segment"]],
        columns=[col_proportion, col_source, col_group]
    ).sort_values(by=col_proportion, ascending=False)

    # stacked by Source
    domain = ["Well Predicted", "Safe marked as Leavers", "Missed Leavers"]
    range_colors = ["#32a838", "#e8911e", "#e8251e"]
    chart = (
        alt.Chart(df_zoom)
        .mark_bar(size=100)
        .encode(
            x=alt.X(f"{col_group}:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{col_proportion}:Q"),
            color=alt.Color(f"{col_source}:N", scale=alt.Scale(domain=domain, range=range_colors)),
            tooltip=alt.Tooltip([col_proportion, col_source]),
        ).properties(
            width=300,
            height=400,
            title="Zoom on algo predictions"
        ).interactive()
    )
    st.altair_chart(chart)

# ===================RISK BY SEGMENT ===================
# segments P1: department, tenure
st.subheader("Risk by segment", divider=True)

# department
departments = df_test["department_fmt"].unique().tolist()
departments = sorted(departments)
departments

# segments
segments_to_select = segments.keys()
select_segment = st.selectbox(
    label="Select the segment to analyze",
    options=segments_to_select

)


def turnover_by_segment(segment_enum, df):
    # eg segment_str = Department, segment_enum = class Dept

    # loop through

    # count
    # segment_group = df.groupby([segment_enum._col_name, "left"])[segment_enum._col_name, "left"].agg( {"left": "sum", segment_enum._col_name: "count"})

    # segment_group = df.groupby([segment_enum._col_name, "left"]).agg(
    #   {"left": "count", segment_enum._col_name: "count"})

    # isolate group by multi-index series with counts. Cf. L0 +++
    gb = df.groupby([segment_enum._col_name, "left"]).size()
    # create df with counts only
    df_segment = gb.to_frame("count")
    # add proportions using the multi-index series of counts -> degroup up to the salary group level then get sum()
    df_segment["proportion"] = round(gb / gb.groupby(level=0).sum() * 100, 2)
    df_segment = df_segment.reset_index()

    # keep turnover data only
    df_turnover = df_segment[df_segment["left"] == 1]
    df_turnover["mean_segment"] = df_turnover["proportion"].sum() / len(df_turnover)

    # compute delta compared to average segment turnover
    df_turnover["delta_prc"] = round(-(1 - df_turnover["proportion"] / df_turnover["mean_segment"]) * 100, 2)

    return df_turnover


def build_etiquette_style(risky):
    color = "red" if risky else "green"
    etiquette_body = """
                border: 2px solid;
                border-color: {border_color};
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 2px;
"""
    etiquette_body = etiquette_body.format(border_color=color)

    segment_cat_body = """
                font-weight: bold;
                color: {cat_color};
    """
    segment_cat_body = segment_cat_body.format(cat_color=color)

    start = """
        <style>
            .etiquette_risky {
""" if risky else """
        <style>
            .etiquette_safe {
"""
    cat_label = """
            }
            .segment_cat_risky {
    """ if risky else """
            }
            .segment_cat_safe {
    """
    end = """
            }
        </style>
    """

    return start + etiquette_body + cat_label + segment_cat_body + end


def format_etiquette(segment_cat, delta):
    #
    risky  = delta > 0
    if risky:
        # risky
        comp = "more"
        segment_cat_class = "segment_cat_risky"
        etiquette_class = "etiquette_risky"
    else:
        # safe
        comp = "less"
        delta = -delta

        segment_cat_class = "segment_cat_safe"
        etiquette_class = "etiquette_safe"
    style = build_etiquette_style( risky=risky)

    # add text in html
    etiquette_txt = f"""
<div class= '{etiquette_class}'>
    <p class='{segment_cat_class}'>{segment_cat}</p
    <p>{delta}% {comp} than the average turnover rate</p>
</div>
"""
    etiquette_fmt = style + etiquette_txt
    return etiquette_fmt

def format_etiquettes(tr_by_segment, segment_enum, col_delta):
    """

    :return:
    """
    col_segment = segment_enum._col_name
    df_cols = ["category", "delta_prc", "etiquette"]
    df_etiquettes = pd.DataFrame(columns=df_cols)
    df_list = list()
    for segment in segment_enum:
        # get category and delta value
        category = segment.formatted
        delta = tr_by_segment.loc[ tr_by_segment[col_segment] == category, col_delta ].values.tolist()[0]
        etiquette = format_etiquette(segment_cat=category, delta=delta)
        df_list.append(pd.DataFrame( [[category, delta, etiquette ]], columns = df_cols, index=[0]))

    # order etiquettes by delta
    df_etiquettes = pd.concat( df_list, axis=0)
    df_etiquettes = df_etiquettes.sort_values(by="delta_prc", ascending=False)
    etiquettes = df_etiquettes["etiquette"].values.tolist()
    return etiquettes

if select_segment:
    # get corresponding enum object
    for s in segments_to_select:
        if select_segment == s:
            # get predicted turnovers by department
            segment_enum = segments[s]
            tr_by_segment = turnover_by_segment(segment_enum=segment_enum, df=df_test)
            #tr_by_segment

            # display turnovers tn by dept
            etiquettes = format_etiquettes(tr_by_segment=tr_by_segment, segment_enum=segment_enum, col_delta="delta_prc")
            nb_etiquettes = len(etiquettes)
            nb_cols = 2
            nb_rows = int(np.ceil(nb_etiquettes/nb_cols))
            # create nb_cols columns nb_rows times
            etiquette_pos = 0
            for row_index in range(nb_rows):
                col1, col2 = st.columns(2)
                col1.markdown(etiquettes[etiquette_pos], unsafe_allow_html=True)
                etiquette_pos += 1
                col2.markdown(etiquettes[etiquette_pos], unsafe_allow_html=True)
                etiquette_pos += 1
        else:
            st.write("Can't find selected segment in mapping")

