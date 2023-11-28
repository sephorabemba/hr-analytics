import streamlit as st

from cache.dataset import load_full_data

st.set_page_config(
    page_title="Employee Turnover Prediction App",
    page_icon="💼",
)

st.sidebar.success("Select a demo from the sidebar")

st.title("🔍 Employee turnover analysis 👩‍💼")

st.write("")

st.subheader("What is this app?", divider=True)

overview = f"""### Overview
This app is a demo of employee turnover prediction
based on the "Hr Analytics Job Prediction" [Kaggle dataset](
https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction).

Select a demo from the sidebar to uncover the story of this data
and **learn how to decrease employee turnover!**
"""

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
* Uncover impactful insights and make recommendations to increase employee retention.

### The approach
The data team has thoroughly analyzed the survey data and has come up with a HR analytics software powered with AI to
solve this business problem (*Use 55 template*).

### Techniques used:
- Exploratory Data Analysis: Python, scikit learn
- Machine Learning modeling: Gradient Boosting algorithm
- Web app development & deployment: Streamlit, Streamlit Cloud

"""

st.markdown(overview)
st.markdown(summary)

# ======= MAIN =======

dict_data = load_full_data()
# load and prepare data
if "dict_data" not in st.session_state:
    st.session_state['dict_data'] = dict_data

st.dataframe(dict_data["df_test"])
