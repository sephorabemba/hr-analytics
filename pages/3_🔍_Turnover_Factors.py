import altair as alt
import streamlit as st

from cache.dataset import load_full_data
from cache.model import get_feat_importance, load_model

st.set_page_config(
    page_title="Turnover Factors",
    page_icon="🔍"
)

dict_data = load_full_data()
if "dict_data" not in st.session_state:
    dict_data = load_full_data()
    st.session_state["dict_data"] = dict_data

pipeline = load_model()

# =================== Turnover Factors ===================
st.subheader("Turnover Factors", divider=True)
st.write("This section highlights the factors that explain employee turnover the most.")
df_importance = get_feat_importance(_pipeline=pipeline)

chart = (
    alt.Chart(df_importance)
    .mark_bar(size=10)
    .encode(
        y=alt.X("Factor:N", axis=alt.Axis(orient="left")).sort("-x"),
        x=alt.Y("Importance (%):Q", scale=alt.Scale(reverse=False)),
        tooltip=alt.Tooltip(["Factor", "Importance (%)"])
    )
    .properties(width=800, height=300, title="Factor Importance")
    .interactive()
)

st.altair_chart(chart)
