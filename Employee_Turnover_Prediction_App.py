import numpy as np
import pandas as pd
import streamlit as st

from cache.dataset import load_full_data
from cache.model import load_model
from utils.html_formatter import format_etiquettes_recos
from utils.utils import (
    Factor,
    recos_filepath,
)

st.set_page_config(
    page_title="Employee Turnover Prediction App",
    page_icon="💼",
)

st.sidebar.success("Select a demo from the sidebar")

st.title("🔍 Employee turnover analysis 👩‍💼")

st.write("Select a demo from the sidebar!")

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


@st.cache_data
def load_recos():
    df_reco = pd.read_csv(recos_filepath)
    return df_reco


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

dict_data = load_full_data()
# load and prepare data
if "dict_data" not in st.session_state:
    st.session_state['dict_data'] = dict_data

df_train, df_valid, df_test = dict_data["df_train"], dict_data["df_valid"], dict_data["df_test"]
X_train, y_train, X_valid, y_valid, X_test, y_test = (
    dict_data["X_train"],
    dict_data["y_train"],
    dict_data["X_valid"],
    dict_data["y_valid"],
    dict_data["X_test"],
    dict_data["y_test"],
)
pipeline = load_model()

st.dataframe(df_test)

# load recommendations
df_reco = load_recos()

# =================== Recommendations ===================

st.subheader("Recommendations", divider=True)
st.write(
    "These recommendations aim at supporting data-driven decisions to "
    "reduce employee turnover. Generated with ChatGPT."
)

factors_to_select = sorted(Factor.get_factor_formatted())
factors_selected = st.multiselect(
    label="Choose 1 or more factors",
    options=factors_to_select,
    default=None,
)

if factors_selected:
    # df_reco
    # get factor ids
    factor_ids = Factor.ids_from_formatted(factors_selected)
    # select targeted reco lines only
    mask = df_reco["id_factor"].isin(factor_ids)
    reco_selected = df_reco[mask]
    reco_selected["factor_fmt"] = reco_selected["id_factor"].apply(lambda id: Factor.id_to_formatted(id))
    reco_selected

    # format etiquettes for each factor
    etiquettes = format_etiquettes_recos(df_reco=reco_selected, df_importance=df_importance)
    # show etiquettes
    for e in etiquettes:
        st.markdown(e, unsafe_allow_html=True)
