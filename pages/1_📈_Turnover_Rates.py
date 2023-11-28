import altair as alt
import pandas as pd
import streamlit as st

from cache.dataset import load_full_data
from cache.model import predict, load_model
from utils.turnover_calculator import compute_turnover

st.set_page_config(
    page_title="Turnover Rates",
    page_icon="📈"
)

dict_data = load_full_data()
if "dict_data" not in st.session_state:
    dict_data = load_full_data()
    st.session_state["dict_data"] = dict_data

pipeline = load_model()

X_train = dict_data["X_train"]
y_train = dict_data["y_train"]
X_test = dict_data["X_test"]
y_test = dict_data["y_test"]

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
